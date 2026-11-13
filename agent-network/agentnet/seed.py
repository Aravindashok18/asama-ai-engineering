"""Load the sample company agents from data/agents.json (used by the demo, eval and tests)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .config import Settings
from .embeddings import Embedder
from .models import AgentRegister
from .service import Platform
from .store import Store

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_profiles(path: Optional[Path] = None) -> list:
    return json.loads((path or DATA_DIR / "agents.json").read_text(encoding="utf-8"))


def seed_platform(platform: Platform, verify: bool = True, path: Optional[Path] = None) -> Dict[str, dict]:
    """Registers every sample agent; returns handle -> {agent, api_key, playbook}."""
    out = {}
    for entry in load_profiles(path):
        reg = platform.register_agent(AgentRegister(**entry["profile"]))
        if verify:
            platform.admin_verify(reg["agent"]["id"])
        out[entry["profile"]["handle"]] = {
            "agent": platform.store.get_agent(reg["agent"]["id"]),
            "api_key": reg["api_key"],
            "playbook": entry.get("playbook", {}),
        }
    return out


def in_memory_platform(embedder: Optional[Embedder] = None, **overrides) -> Platform:
    kwargs = {"db_path": ":memory:", "background_worker": False, "exploration_rate": 0.0}
    if embedder is not None:
        kwargs["embedder"] = embedder.name
    kwargs.update(overrides)
    return Platform(settings=Settings(**kwargs), store=Store(":memory:"), embedder=embedder)
