"""Notification delivery.

Notifications are durable rows and every agent can always pull them from its inbox
(GET /v1/notifications). A webhook is a best-effort push on top: signed, retried with
exponential backoff, and paused per agent by a circuit breaker when the endpoint keeps
failing, so one slow partner cannot hold up the queue for everyone else.
"""
from __future__ import annotations

import json
from typing import Callable, Dict, Optional

import httpx

from .config import Settings
from .safety import sign_payload
from .store import Store, now


class Deliverer:
    def __init__(self, store: Store, settings: Settings, payload_for: Callable[[dict], dict],
                 client: Optional[httpx.Client] = None):
        self.store = store
        self.settings = settings
        self.payload_for = payload_for
        self.client = client or httpx.Client(timeout=settings.webhook_timeout_s, follow_redirects=False)

    def deliver_due(self, limit: int = 200, at: Optional[float] = None) -> Dict[str, int]:
        at = now() if at is None else at
        stats = {"delivered": 0, "retrying": 0, "failed": 0, "paused": 0}
        for n in self.store.due_webhooks(at, limit):
            agent = self.store.get_agent(n["agent_id"])
            if agent is None or agent["status"] != "active" or not agent["webhook_url"]:
                self.store.update_notification(n["id"], status="failed", last_error="agent inactive or no webhook")
                stats["failed"] += 1
                continue
            if agent["webhook_paused_until"] > at:
                self.store.update_notification(n["id"], next_attempt_at=agent["webhook_paused_until"])
                stats["paused"] += 1
                continue
            stats[self._send(n, agent, at)] += 1
        return stats

    def _send(self, n: dict, agent: dict, at: float) -> str:
        body = json.dumps(self.payload_for(n), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ts, signature = sign_payload(agent["webhook_secret"], body, int(at))
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AgentNet-Webhooks/1.0",
            "X-AgentNet-Delivery": n["id"],  # idempotency key for the receiver
            "X-AgentNet-Timestamp": ts,
            "X-AgentNet-Signature": signature,
        }
        error = None
        try:
            resp = self.client.post(agent["webhook_url"], content=body, headers=headers)
            if 200 <= resp.status_code < 300:
                self.store.update_notification(n["id"], status="delivered", delivered_at=at,
                                               attempts=n["attempts"] + 1, last_error=None)
                if agent["consecutive_failures"]:
                    self.store.update_agent(agent["id"], consecutive_failures=0)
                return "delivered"
            error = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]

        attempts = n["attempts"] + 1
        failures = agent["consecutive_failures"] + 1
        agent_update = {"consecutive_failures": failures}
        if failures >= self.settings.circuit_breaker_failures:
            agent_update["webhook_paused_until"] = at + self.settings.circuit_breaker_pause_s
        self.store.update_agent(agent["id"], **agent_update)

        if attempts >= self.settings.webhook_max_attempts:
            self.store.update_notification(n["id"], status="failed", attempts=attempts, last_error=error)
            return "failed"
        backoff = self.settings.webhook_backoff_base_s * (2 ** (attempts - 1))
        self.store.update_notification(n["id"], attempts=attempts, last_error=error, next_attempt_at=at + backoff)
        return "retrying"
