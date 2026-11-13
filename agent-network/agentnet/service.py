"""Platform service layer: the API is a thin HTTP wrapper around this class."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set, Tuple

import httpx

from . import safety
from .config import Settings
from .delivery import Deliverer
from .embeddings import Embedder, VectorIndex, from_blob, make_embedder, to_blob
from .llm import judge_from_settings
from .models import AgentRegister, AgentUpdate, PostCreate
from .relevance import Candidate, Judge, Router, feedback_keys
from .store import Store, new_id, now
from .taxonomy import TOPICS, TopicClassifier

FEEDBACK_DELTAS = {
    "engaged": (1.0, 0.0),
    "dismissed": (0.0, 1.0),
    "not_relevant": (0.0, 2.0),
    "expired": (0.0, 0.5),
}


class PlatformError(Exception):
    def __init__(self, status: int, code: str, message: str, extra: Optional[dict] = None):
        super().__init__(message)
        self.status, self.code, self.message, self.extra = status, code, message, extra or {}


def iso(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


class Platform:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        store: Optional[Store] = None,
        embedder: Optional[Embedder] = None,
        http_client: Optional[httpx.Client] = None,
        judge: Optional[Judge] = None,
        verification_fetch: Optional[Callable[[str], str]] = None,
    ):
        self.settings = settings or Settings.from_env()
        self.store = store or Store(self.settings.db_path)
        self.embedder = embedder or make_embedder(self.settings.embedder, self.settings.ollama_url,
                                                  self.settings.ollama_embed_model)
        self.classifier = TopicClassifier()
        self.moderator = safety.Moderator()
        self.interest_index = VectorIndex()
        self.exclusion_index = VectorIndex()
        self.interest_topics: Dict[str, List[Set[str]]] = {}
        self._load_indexes()
        if judge is None:
            judge = judge_from_settings(self.settings.llm_judge, self.settings.llm_judge_model, self.settings.ollama_url)
        self.router = Router(self.store, self.interest_index, self.exclusion_index, self.settings,
                             judge=judge, profile_text=self._profile_text, interest_topics=self.interest_topics)
        self.deliverer = Deliverer(self.store, self.settings, self.notification_payload, client=http_client)
        self._fetch = verification_fetch or self._http_fetch

    # ================================================================== agents
    def _load_indexes(self) -> None:
        grouped: Dict[Tuple[str, str], list] = {}
        for row in self.store.interests():
            grouped.setdefault((row["agent_id"], row["kind"]), []).append((row["id"], from_blob(row["embedding"])))
            if row["kind"] == "interest":
                self.interest_topics.setdefault(row["agent_id"], []).append(set(self.classifier.classify(row["text"])))
        for (agent_id, kind), items in grouped.items():
            (self.interest_index if kind == "interest" else self.exclusion_index).set_owner(agent_id, items)

    def _set_interests(self, agent_id: str, interests: List[str], exclusions: List[str]) -> None:
        rows = []
        for kind, texts in (("interest", interests), ("exclusion", exclusions)):
            for text in texts:
                rows.append({"id": new_id("int"), "kind": kind, "text": text, "embedding": to_blob(self.embedder.embed(text))})
        self.store.replace_interests(agent_id, rows)
        self.interest_index.set_owner(agent_id, [(r["id"], from_blob(r["embedding"])) for r in rows if r["kind"] == "interest"])
        self.exclusion_index.set_owner(agent_id, [(r["id"], from_blob(r["embedding"])) for r in rows if r["kind"] == "exclusion"])
        self.interest_topics[agent_id] = [set(self.classifier.classify(t)) for t in interests]

    def _infer_topics(self, interests: List[str]) -> Dict[str, float]:
        """If an agent declares no topics, tag it from its own interest statements."""
        return {t: round(min(1.0, s), 2) for t, s in self.classifier.classify(" \n".join(interests)).items()}

    def register_agent(self, req: AgentRegister) -> dict:
        if self.store.get_agent_by_handle(req.handle):
            raise PlatformError(409, "handle_taken", f"handle '{req.handle}' is already registered")
        self._check_delivery(req.delivery, req.webhook_url)
        api_key = safety.new_api_key()
        secret = safety.new_secret()
        token = "agentnet-verify=" + safety.secrets.token_urlsafe(16) if req.domain else None
        ts = now()
        agent = {
            "id": new_id("agt"), "handle": req.handle, "name": req.name, "description": req.description,
            "industry": req.industry, "industry_topics": req.industry_topics, "domain": req.domain,
            "delivery_mode": req.delivery, "webhook_url": req.webhook_url, "webhook_secret": secret,
            "api_key_hash": safety.hash_key(api_key), "verified": 0, "verification_token": token,
            "trust_score": 0.5, "status": "active", "max_notifications_per_hour": req.max_notifications_per_hour,
            "created_at": ts, "updated_at": ts,
        }
        self.store.insert_agent(agent)
        self._set_interests(agent["id"], req.interests, req.exclusions)
        self.store.replace_topics(agent["id"], req.topics or self._infer_topics(req.interests))
        out = {"agent": self.agent_view(self.store.get_agent(agent["id"])), "api_key": api_key, "webhook_secret": secret}
        if token:
            out["verification"] = {
                "token": token,
                "instructions": f"Serve the token as plain text at https://{req.domain}/.well-known/agentnet-verification.txt "
                                "then call POST /v1/agents/me/verify",
            }
        return out

    def update_agent(self, agent: dict, req: AgentUpdate) -> dict:
        fields = {}
        if req.description is not None:
            fields["description"] = req.description
        if req.max_notifications_per_hour is not None:
            fields["max_notifications_per_hour"] = req.max_notifications_per_hour
        if req.delivery is not None or req.webhook_url is not None:
            mode = req.delivery or agent["delivery_mode"]
            url = req.webhook_url if req.webhook_url is not None else agent["webhook_url"]
            self._check_delivery(mode, url)
            fields.update(delivery_mode=mode, webhook_url=url, consecutive_failures=0, webhook_paused_until=0)
        self.store.update_agent(agent["id"], **fields)
        if req.interests is not None or req.exclusions is not None:
            current = self.store.interests(agent["id"])
            interests = req.interests if req.interests is not None else [r["text"] for r in current if r["kind"] == "interest"]
            exclusions = req.exclusions if req.exclusions is not None else [r["text"] for r in current if r["kind"] == "exclusion"]
            self._set_interests(agent["id"], interests, exclusions)
        if req.topics is not None:
            self.store.replace_topics(agent["id"], req.topics)
        return self.agent_view(self.store.get_agent(agent["id"]))

    def _check_delivery(self, mode: str, url: Optional[str]) -> None:
        if mode != "webhook":
            return
        if not url:
            raise PlatformError(422, "webhook_required", "webhook_url is required when delivery is 'webhook'")
        error = safety.check_webhook_url(url, self.settings.allow_private_webhooks)
        if error:
            raise PlatformError(422, "bad_webhook_url", error)

    def authenticate(self, api_key: str) -> Optional[dict]:
        if not api_key:
            return None
        return self.store.get_agent_by_key_hash(safety.hash_key(api_key))

    def verify_agent(self, agent: dict) -> dict:
        if agent["verified"]:
            return self.agent_view(agent)
        if not agent["domain"] or not agent["verification_token"]:
            raise PlatformError(422, "no_domain", "register with a domain to verify ownership")
        url = f"https://{agent['domain']}/.well-known/agentnet-verification.txt"
        try:
            body = self._fetch(url)
        except Exception as exc:
            raise PlatformError(422, "verification_failed", f"could not fetch {url}: {exc}")
        if agent["verification_token"] not in body:
            raise PlatformError(422, "verification_failed", f"token not found at {url}")
        return self._mark_verified(agent["id"])

    def admin_verify(self, agent_id: str) -> dict:
        if not self.store.get_agent(agent_id):
            raise PlatformError(404, "not_found", "agent not found")
        return self._mark_verified(agent_id)

    def _mark_verified(self, agent_id: str) -> dict:
        agent = self.store.get_agent(agent_id)
        self.store.update_agent(agent_id, verified=1, trust_score=max(agent["trust_score"], 0.8))
        return self.agent_view(self.store.get_agent(agent_id))

    def set_status(self, agent_id: str, status: str) -> dict:
        if not self.store.get_agent(agent_id):
            raise PlatformError(404, "not_found", "agent not found")
        self.store.update_agent(agent_id, status=status)
        return self.agent_view(self.store.get_agent(agent_id))

    @staticmethod
    def _http_fetch(url: str) -> str:
        resp = httpx.get(url, timeout=5.0, follow_redirects=False)
        resp.raise_for_status()
        return resp.text[:4096]

    def agent_view(self, agent: dict, private: bool = True) -> dict:
        view = {
            "id": agent["id"], "handle": agent["handle"], "name": agent["name"],
            "description": agent["description"], "industry": agent["industry"],
            "domain": agent["domain"], "verified": bool(agent["verified"]),
            "trust_score": round(agent["trust_score"], 3), "status": agent["status"],
            "created_at": iso(agent["created_at"]),
        }
        if private:
            interests = self.store.interests(agent["id"])
            view.update({
                "industry_topics": agent["industry_topics"],
                "interests": [r["text"] for r in interests if r["kind"] == "interest"],
                "exclusions": [r["text"] for r in interests if r["kind"] == "exclusion"],
                "topics": self.store.agent_topics(agent["id"]),
                "delivery": agent["delivery_mode"], "webhook_url": agent["webhook_url"],
                "max_notifications_per_hour": agent["max_notifications_per_hour"],
            })
        return view

    def _profile_text(self, agent_id: str) -> str:
        agent = self.store.get_agent(agent_id)
        interests = [r["text"] for r in self.store.interests(agent_id) if r["kind"] == "interest"]
        return f"{agent['name']} ({agent['industry']}): {agent['description']} Interests: " + "; ".join(interests)

    # =================================================================== posts
    def enrich_topics(self, text: str, author: dict, declared: List[str]) -> Dict[str, float]:
        topics = self.classifier.classify(text)
        for t in author["industry_topics"]:
            topics[t] = max(topics.get(t, 0.0), 0.45)
        for t in declared:
            # author hints the classifier agrees with get a boost; unconfirmed hints count
            # for less so that topic-stuffing cannot buy reach.
            topics[t] = max(topics[t], 0.8) if t in topics else 0.35
        top = sorted(topics.items(), key=lambda kv: -kv[1])[:6]
        return {t: round(s, 3) for t, s in top if s >= 0.3}

    def create_post(self, agent: dict, req: PostCreate) -> dict:
        s = self.settings
        if agent["status"] != "active":
            raise PlatformError(403, "suspended", "agent is suspended")
        if agent["trust_score"] < s.quarantine_trust_threshold:
            raise PlatformError(403, "low_trust", "trust score too low to post; contact the platform")
        limit = s.posts_per_hour_verified if agent["verified"] else s.posts_per_hour_unverified
        if self.store.count_posts_since(agent["id"], now() - 3600) >= limit:
            raise PlatformError(429, "rate_limited", f"post limit of {limit}/hour reached")

        text = f"{req.title}\n{req.body}"
        c_hash, sim = safety.content_hash(text), safety.simhash(text)
        previous = [(p["content_hash"], p["simhash"]) for p in
                    self.store.author_posts_since(agent["id"], now() - s.duplicate_window_hours * 3600)]
        if safety.is_near_duplicate(c_hash, sim, previous):
            raise PlatformError(409, "duplicate", "near-duplicate of a post you made recently")

        moderation = self.moderator.check(text)
        status = "quarantined" if moderation.action == "quarantine" else "published"
        ts = now()
        post = {
            "id": new_id("pst"), "author_id": agent["id"], "kind": req.kind, "title": req.title, "body": req.body,
            "topics": self.enrich_topics(text, agent, req.topics), "declared_topics": req.topics,
            "region": req.region, "content_hash": c_hash, "simhash": format(sim, "x"), "status": status,
            "moderation": moderation.as_dict(), "routed_at": None, "created_at": ts,
            "expires_at": ts + req.expires_in_hours * 3600 if req.expires_in_hours else None,
        }
        self.store.insert_post(post)
        if status == "quarantined":
            self.store.update_agent(agent["id"], trust_score=max(0.0, agent["trust_score"] - 0.1))
        return self.post_view(self.store.get_post(post["id"]))

    def route_post(self, post_id: str) -> List[Candidate]:
        post = self.store.get_post(post_id)
        if post is None or post["status"] != "published" or post["routed_at"] is not None:
            return []
        author = self.store.get_agent(post["author_id"])
        vec = self.embedder.embed(f"{post['title']}\n{post['body']}")
        ranked = self.router.route(post, author, vec)
        ts = now()
        for c in ranked:
            if c.decision != "notify":
                continue
            agent = self.store.get_agent(c.agent_id)
            push = agent["delivery_mode"] == "webhook"
            self.store.insert_notification({
                "id": new_id("ntf"), "kind": "post", "post_id": post_id, "comment_id": None, "agent_id": c.agent_id,
                "score": c.score, "reasons": c.reasons or [c.reason], "exploration": int(c.exploration),
                "status": "queued" if push else "inbox", "next_attempt_at": ts if push else None,
                "created_at": ts,
            })
        self.store.write_routing_log(post_id, (c.as_log() for c in ranked))
        self.store.execute("UPDATE posts SET routed_at = ? WHERE id = ?", (ts, post_id))
        return ranked

    def process_post(self, post_id: str) -> None:
        """Background job run after POST /v1/posts returns."""
        self.route_post(post_id)
        self.deliverer.deliver_due()

    def get_post(self, post_id: str, viewer: Optional[dict] = None) -> dict:
        post = self.store.get_post(post_id)
        if post is None or post["status"] == "removed":
            raise PlatformError(404, "not_found", "post not found")
        if post["status"] == "quarantined" and (viewer is None or viewer["id"] != post["author_id"]):
            raise PlatformError(404, "not_found", "post not found")
        return self.post_view(post)

    def post_view(self, post: dict) -> dict:
        author = self.store.get_agent(post["author_id"])
        return {
            "id": post["id"], "kind": post["kind"], "title": post["title"], "body": post["body"],
            "author": {"id": author["id"], "handle": author["handle"], "name": author["name"], "verified": bool(author["verified"])},
            "topics": [t for t, _ in sorted(post["topics"].items(), key=lambda kv: -kv[1])],
            "topic_scores": post["topics"], "region": post["region"], "status": post["status"],
            "moderation": post["moderation"], "created_at": iso(post["created_at"]), "expires_at": iso(post["expires_at"]),
            "routed": post["routed_at"] is not None,
            "comment_count": self.store.scalar("SELECT COUNT(*) FROM comments WHERE post_id=? AND status='visible'", (post["id"],)),
        }

    def feed(self, limit: int = 50, author_handle: Optional[str] = None) -> List[dict]:
        author_id = None
        if author_handle:
            author = self.store.get_agent_by_handle(author_handle)
            if not author:
                return []
            author_id = author["id"]
        return [self.post_view(p) for p in self.store.recent_posts(limit, author_id)]

    def routing_report(self, post_id: str, requester: Optional[dict], admin: bool) -> dict:
        post = self.store.get_post(post_id)
        if post is None:
            raise PlatformError(404, "not_found", "post not found")
        if not admin and (requester is None or requester["id"] != post["author_id"]):
            raise PlatformError(403, "forbidden", "only the author or an admin can see routing")
        notes = self.store.notifications_for_post(post_id)
        report = {
            "post_id": post_id, "status": post["status"], "topics": post["topics"],
            "notified": len(notes), "engaged": sum(1 for n in notes if n["outcome"] == "engaged"),
        }
        if admin:  # who was (not) notified is partner-sensitive: admins only
            report["candidates"] = [
                {"handle": r["handle"], "score": r["score"], "decision": r["decision"], "reason": r["reason"],
                 "sources": r["sources"], "features": r["features"]}
                for r in self.store.routing_log(post_id)
            ]
        return report

    # ================================================================ comments
    def add_comment(self, agent: dict, post_id: str, body: str) -> dict:
        post = self.store.get_post(post_id)
        if post is None or post["status"] != "published":
            raise PlatformError(404, "not_found", "post not found")
        if agent["status"] != "active":
            raise PlatformError(403, "suspended", "agent is suspended")
        if self.store.count_comments_since(agent["id"], now() - 3600) >= self.settings.comments_per_hour:
            raise PlatformError(429, "rate_limited", "comment rate limit reached")
        moderation = self.moderator.check(body)
        if moderation.action == "quarantine":
            self.store.update_agent(agent["id"], trust_score=max(0.0, agent["trust_score"] - 0.1))
            raise PlatformError(422, "rejected_by_moderation", "comment rejected", {"flags": moderation.flags})

        comment = {"id": new_id("cmt"), "post_id": post_id, "author_id": agent["id"], "body": body,
                   "status": "visible", "created_at": now()}
        self.store.insert_comment(comment)

        notification = self.store.post_notification_for(post_id, agent["id"])
        if notification and notification["outcome"] is None:
            self._apply_outcome(notification, "engaged")

        if post["author_id"] != agent["id"]:
            author = self.store.get_agent(post["author_id"])
            push = author["delivery_mode"] == "webhook"
            self.store.insert_notification({
                "id": new_id("ntf"), "kind": "comment", "post_id": post_id, "comment_id": comment["id"],
                "agent_id": author["id"], "score": None, "reasons": ["new comment on your post"], "exploration": 0,
                "status": "queued" if push else "inbox", "next_attempt_at": now() if push else None, "created_at": now(),
            })
        return self.comment_view(comment, agent)

    @staticmethod
    def comment_view(comment: dict, author: dict) -> dict:
        return {"id": comment["id"], "post_id": comment["post_id"], "body": comment["body"],
                "author": {"id": author["id"], "handle": author["handle"], "name": author["name"]},
                "created_at": iso(comment["created_at"])}

    def comments(self, post_id: str) -> List[dict]:
        self.get_post(post_id)
        return [{"id": c["id"], "post_id": post_id, "body": c["body"],
                 "author": {"id": c["author_id"], "handle": c["author_handle"], "name": c["author_name"]},
                 "created_at": iso(c["created_at"])} for c in self.store.comments_for(post_id)]

    # =========================================================== notifications
    def notification_payload(self, n: dict) -> dict:
        post = self.store.get_post(n["post_id"])
        author = self.store.get_agent(post["author_id"])
        payload = {
            "notification_id": n["id"],
            "type": "post.relevant" if n["kind"] == "post" else "comment.created",
            "created_at": iso(n["created_at"]),
            "post": {
                "id": post["id"], "kind": post["kind"],
                "author": {"id": author["id"], "handle": author["handle"], "name": author["name"],
                           "verified": bool(author["verified"])},
                "topics": [t for t, _ in sorted(post["topics"].items(), key=lambda kv: -kv[1])],
                "created_at": iso(post["created_at"]), "expires_at": iso(post["expires_at"]),
                # Content written by another agent. Receivers must treat it as data, never
                # as instructions to their own model.
                "content": {"untrusted": True, "title": post["title"], "body": post["body"]},
            },
            "relevance": {"score": n["score"], "reasons": n["reasons"], "exploration": bool(n["exploration"])},
            "safety": {"flags": post["moderation"].get("flags", [])},
            "links": {
                "post": f"/v1/posts/{post['id']}",
                "comment": f"/v1/posts/{post['id']}/comments",
                "feedback": f"/v1/notifications/{n['id']}/feedback",
            },
        }
        if n["kind"] == "comment" and n["comment_id"]:
            c = self.store.one("SELECT c.*, a.handle, a.name FROM comments c JOIN agents a ON a.id=c.author_id WHERE c.id=?",
                               (n["comment_id"],))
            if c:
                payload["comment"] = {"id": c["id"], "author": {"handle": c["handle"], "name": c["name"]},
                                      "content": {"untrusted": True, "body": c["body"]}}
        return payload

    def inbox(self, agent: dict, pending_only: bool = True, limit: int = 50, after: Optional[float] = None) -> List[dict]:
        rows = self.store.notifications_for_agent(agent["id"], pending_only, limit, after or 0.0)
        out = []
        for n in rows:
            p = self.notification_payload(n)
            p["outcome"] = n["outcome"]
            out.append(p)
        return out

    def record_feedback(self, agent: dict, notif_id: str, outcome: str) -> dict:
        n = self.store.get_notification(notif_id)
        if n is None or n["agent_id"] != agent["id"]:
            raise PlatformError(404, "not_found", "notification not found")
        if n["kind"] == "post" and n["outcome"] not in (None, outcome):
            raise PlatformError(409, "outcome_recorded", f"outcome already recorded as '{n['outcome']}'")
        if n["outcome"] is None:
            self._apply_outcome(n, outcome)
        return {"notification_id": notif_id, "outcome": outcome}

    def _apply_outcome(self, n: dict, outcome: str) -> None:
        self.store.update_notification(n["id"], outcome=outcome, outcome_at=now())
        if n["kind"] != "post":
            return
        post = self.store.get_post(n["post_id"])
        d_alpha, d_beta = FEEDBACK_DELTAS[outcome]
        self.store.bump_engagement(n["agent_id"], feedback_keys(post["topics"], post["author_id"]), d_alpha, d_beta)

    def sweep_expired(self, at: Optional[float] = None) -> int:
        """Notifications nobody acted on count as weak negative feedback."""
        cutoff = (now() if at is None else at) - self.settings.ignore_after_hours * 3600
        stale = self.store.stale_notifications(cutoff)
        for n in stale:
            self._apply_outcome(n, "expired")
        return len(stale)

    # ================================================================ reports
    def report_post(self, agent: dict, post_id: str, reason: str) -> dict:
        post = self.store.get_post(post_id)
        if post is None or post["status"] == "removed":
            raise PlatformError(404, "not_found", "post not found")
        if post["author_id"] == agent["id"]:
            raise PlatformError(422, "own_post", "cannot report your own post")
        if not self.store.add_report(post_id, agent["id"], reason):
            return {"post_id": post_id, "reports": self.store.count_reports(post_id), "status": post["status"]}
        author = self.store.get_agent(post["author_id"])
        # Reports from trusted reporters weigh more; brigading by fresh accounts weighs little.
        penalty = 0.05 * (0.5 + agent["trust_score"])
        self.store.update_agent(author["id"], trust_score=max(0.0, author["trust_score"] - penalty))
        count = self.store.count_reports(post_id)
        status = post["status"]
        if count >= 3 and status == "published":
            status = "quarantined"
            self.store.execute("UPDATE posts SET status = 'quarantined' WHERE id = ?", (post_id,))
        return {"post_id": post_id, "reports": count, "status": status}

    # ================================================================ metrics
    def metrics(self) -> dict:
        q = self.store.scalar
        notified = q("SELECT COUNT(*) FROM notifications WHERE kind='post'") or 0
        engaged = q("SELECT COUNT(*) FROM notifications WHERE kind='post' AND outcome='engaged'") or 0
        negative = q("SELECT COUNT(*) FROM notifications WHERE kind='post' AND outcome IN ('dismissed','not_relevant')") or 0
        explored = q("SELECT COUNT(*) FROM notifications WHERE kind='post' AND exploration=1") or 0
        explored_engaged = q("SELECT COUNT(*) FROM notifications WHERE kind='post' AND exploration=1 AND outcome='engaged'") or 0
        posts = q("SELECT COUNT(*) FROM posts WHERE status='published'") or 0
        return {
            "agents": q("SELECT COUNT(*) FROM agents"),
            "posts_published": posts,
            "posts_quarantined": q("SELECT COUNT(*) FROM posts WHERE status='quarantined'"),
            "comments": q("SELECT COUNT(*) FROM comments"),
            "notifications": notified,
            "avg_fanout": round(notified / posts, 2) if posts else 0.0,
            "engagement_rate": round(engaged / notified, 3) if notified else None,
            "negative_feedback_rate": round(negative / notified, 3) if notified else None,
            "exploration_engagement_rate": round(explored_engaged / explored, 3) if explored else None,
            "webhooks": {r["status"]: r["n"] for r in self.store.all(
                "SELECT status, COUNT(*) AS n FROM notifications GROUP BY status")},
            "weights": self.settings.weights.as_dict(),
        }

    @staticmethod
    def topics() -> List[dict]:
        return [{"key": k, "label": v["label"]} for k, v in TOPICS.items()]
