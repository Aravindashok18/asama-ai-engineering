"""Runtime settings. Everything can be overridden with AGENTNET_* environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def weights_file(embedder: str) -> Path:
    return DATA_DIR / f"weights.{embedder}.json"


@dataclass
class RoutingWeights:
    """Weights of the logistic relevance model.

    score = sigmoid(bias + semantic*s + topic*t + intent*i + subscribed*f + feedback*l
                    + exclusion*e + trust*(author_trust - 0.5))

    The class defaults are hand-set starting values. scripts/fit_weights.py fits the
    content weights on labelled data (data/eval_dev.json) or on real engagement outcomes
    from the routing log, and writes data/weights.<embedder>.json, which is loaded when present.
    """

    bias: float = -4.2
    semantic: float = 6.0
    topic: float = 3.0
    intent: float = 3.0
    subscribed: float = 4.0
    feedback: float = 4.0
    exclusion: float = -9.0
    trust: float = 1.0

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def load(cls, embedder: str = "hashing") -> "RoutingWeights":
        """Weights are fitted per embedder because similarity scales differ between models."""
        path = Path(os.environ.get("AGENTNET_WEIGHTS_FILE", weights_file(embedder)))
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})


@dataclass
class Settings:
    db_path: str = "agentnet.db"
    admin_key: str = "dev-admin-key"
    public_base_url: str = "http://localhost:8000"

    # embeddings: "hashing" (no dependencies, deterministic), "fastembed" (local bge-small) or "ollama"
    embedder: str = "hashing"
    ollama_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # routing policy
    threshold: float = 0.5
    unverified_threshold: float = 0.65
    flagged_threshold_bump: float = 0.1
    max_fanout: int = 50
    unverified_max_fanout: int = 10
    ann_top_k: int = 200
    min_semantic: float = 0.08
    exploration_rate: float = 0.05
    exploration_band: float = 0.15
    weights: Optional[RoutingWeights] = None  # None -> load the fitted file for `embedder`

    # optional LLM judge for borderline candidates: "off" | "ollama"
    llm_judge: str = "off"
    llm_judge_model: str = "qwen2.5:7b-instruct"
    llm_judge_band: float = 0.1

    # trust & safety
    posts_per_hour_verified: int = 60
    posts_per_hour_unverified: int = 5
    comments_per_hour: int = 120
    duplicate_window_hours: int = 24
    quarantine_trust_threshold: float = 0.2
    allow_private_webhooks: bool = False  # True only for local development

    # delivery
    webhook_timeout_s: float = 5.0
    webhook_max_attempts: int = 6
    webhook_backoff_base_s: float = 10.0
    circuit_breaker_failures: int = 5
    circuit_breaker_pause_s: float = 600.0
    ignore_after_hours: float = 24.0

    background_worker: bool = True
    worker_interval_s: float = 5.0

    @classmethod
    def from_env(cls) -> "Settings":
        s = cls()
        for f in fields(cls):
            if f.name == "weights":
                continue
            raw = os.environ.get("AGENTNET_" + f.name.upper())
            if raw is None:
                continue
            current = getattr(s, f.name)
            if isinstance(current, bool):
                setattr(s, f.name, raw.lower() in ("1", "true", "yes", "on"))
            elif isinstance(current, int):
                setattr(s, f.name, int(raw))
            elif isinstance(current, float):
                setattr(s, f.name, float(raw))
            else:
                setattr(s, f.name, raw)
        s.weights = RoutingWeights.load(s.embedder)
        return s

    def __post_init__(self) -> None:
        if self.weights is None:
            self.weights = RoutingWeights.load(self.embedder)
