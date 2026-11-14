"""Writes site/manifest.json: the source files the web page loads into the browser's Python.

Run after changing either package:  python site/build_manifest.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = [
    "agent-network/agentnet/*.py",
    "agent-network/data/agents.json",
    "agent-network/data/weights.*.json",
    "hardware-assistant/hwqa/*.py",
    "hardware-assistant/fixtures/*/meta.json",
    "hardware-assistant/fixtures/*/commands/*",
    "hardware-assistant/fixtures/*/files/**/*",
    "site/web_demo.py",
]
files = sorted({p.relative_to(ROOT).as_posix() for pat in PATTERNS for p in ROOT.glob(pat) if p.is_file()})
(ROOT / "site" / "manifest.json").write_text(json.dumps(files, indent=1) + "\n")
print(f"{len(files)} files")
