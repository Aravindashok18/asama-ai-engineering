"""Model backends. All speak one internal (OpenAI-style) message format.

- OllamaBackend       local model with native tool calling (default; nothing leaves the machine)
- OpenAICompatBackend llama.cpp server / vLLM / LM Studio locally, or a hosted API with --allow-remote
- RuleBackend         no model at all: keyword intent router + templated answers. Works
                      everywhere, deterministic, and is what the tests exercise.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any


@dataclass
class Turn:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


class BackendError(Exception):
    pass


def _parse_text_tool_call(content: str) -> List[ToolCall]:
    """Small local models sometimes print the call as JSON instead of using tool_calls."""
    m = re.search(r"\{.*\}", content or "", re.S)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return []
    if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj["name"].startswith("get_"):
        return [ToolCall(f"call_{uuid.uuid4().hex[:8]}", obj["name"], obj.get("arguments") or obj.get("parameters") or {})]
    return []


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".localhost")


class OllamaBackend:
    name = "ollama"

    def __init__(self, model: str = "qwen2.5:7b-instruct", base_url: str = "http://localhost:11434",
                 timeout: float = 120.0, allow_remote: bool = False):
        if not is_local_url(base_url) and not allow_remote:
            raise BackendError(f"{base_url} is not on this machine; hardware data would leave the laptop. "
                               "Pass --allow-remote to accept that.")
        self.model = model
        self.local = is_local_url(base_url)
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def available(self) -> bool:
        try:
            tags = self.client.get("/api/tags", timeout=2.0).json()
        except (httpx.HTTPError, ValueError):
            return False
        names = {m.get("name") for m in tags.get("models", [])}
        return self.model in names or f"{self.model}:latest" in names

    def complete(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Turn:
        msgs = []
        for m in messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                    {"function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
                    for c in m["tool_calls"]]})
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "content": m["content"], "tool_name": m.get("name")})
            else:
                msgs.append({"role": m["role"], "content": m["content"]})
        body = {"model": self.model, "messages": msgs, "stream": False, "options": {"temperature": 0}}
        if tools:
            body["tools"] = tools
        try:
            resp = self.client.post("/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc
        msg = resp.json().get("message", {})
        calls = [ToolCall(f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], c["function"].get("arguments") or {})
                 for c in msg.get("tool_calls") or []]
        content = msg.get("content") or ""
        if not calls and tools:
            calls = _parse_text_tool_call(content)
        return Turn(content="" if calls else content.strip(), tool_calls=calls)


class OpenAICompatBackend:
    name = "openai-compatible"

    def __init__(self, model: str, base_url: str = "http://localhost:8080/v1", api_key: str = "",
                 timeout: float = 120.0, allow_remote: bool = False):
        if not is_local_url(base_url) and not allow_remote:
            raise BackendError(f"{base_url} is not on this machine; hardware data would leave the laptop. "
                               "Pass --allow-remote to accept that.")
        self.model = model
        self.local = is_local_url(base_url)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)

    def complete(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Turn:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            body["tools"] = tools
        try:
            resp = self.client.post("/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise BackendError(f"model request failed: {exc}") from exc
        msg = resp.json()["choices"][0]["message"]
        calls = []
        for c in msg.get("tool_calls") or []:
            raw = c["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except ValueError:
                args = raw  # validation will reject it and tell the model
            calls.append(ToolCall(c.get("id") or f"call_{uuid.uuid4().hex[:8]}", c["function"]["name"], args))
        content = msg.get("content") or ""
        if not calls and tools:
            calls = _parse_text_tool_call(content)
        return Turn(content="" if calls else content.strip(), tool_calls=calls)


# =========================================================== rule-based backend

TOPICS = [
    ("memory", r"\b(ram|memory|dimm|sodimm|so-dimm|ddr\d?)\b", ["get_memory_info"]),
    ("disk_usage", r"\b(space|free|full|usage|used|left|remaining)\b.*\b(disk|drive|ssd|storage|partition|root|home)\b|"
                   r"\b(disk|drive|ssd|storage)\b.*\b(space|full|left|remaining)\b", ["get_disk_usage"]),
    ("storage", r"\b(ssd|nvme|hdd|hard ?drive|disk|disks|drive|drives|storage|sata|m\.2|partition)\b", ["get_storage_info"]),
    ("gpu", r"\b(gpu|gpus|graphics|video card|nvidia|radeon|geforce|vram|cuda|display adapter)\b", ["get_gpu_info"]),
    ("cpu", r"\b(cpu|processor|cores?|threads?|ghz|clock|avx\w*|ryzen|intel core|virtuali[sz]ation)\b", ["get_cpu_info"]),
    ("battery", r"\b(battery|charge|charging|cycles?|power adapter|plugged)\b", ["get_battery_info"]),
    ("network", r"\b(wi-?fi|wireless|ethernet|network|lan|nic|internet adapter)\b", ["get_network_adapters"]),
    ("temperature", r"\b(temp|temps|temperature|temperatures|hot|heat|thermal|overheat\w*)\b", ["get_temperatures"]),
    ("usb", r"\b(usb|webcam|camera|bluetooth|fingerprint)\b", ["get_usb_devices"]),
    ("system", r"\b(model|make|manufacturer|brand|bios|uefi|firmware|os|distro|distribution|kernel|linux version|"
               r"which laptop|what laptop|motherboard)\b", ["get_system_info"]),
]
OVERVIEW = r"\b(specs?|specifications|summary|overview|everything|about (my|this) (laptop|computer|machine|pc))\b"
OVERVIEW_TOOLS = ["get_system_info", "get_cpu_info", "get_memory_info", "get_storage_info", "get_gpu_info"]
UPGRADE = r"\b(upgrade|add|more|expand|increase|max(imum)?|slots?|support|extra)\b"


def detect_tools(question: str) -> List[str]:
    q = question.lower()
    if re.search(OVERVIEW, q):
        return list(OVERVIEW_TOOLS)
    tools: List[str] = []
    for topic, pattern, names in TOPICS:
        if re.search(pattern, q):
            if topic == "storage" and "get_disk_usage" in tools:
                continue
            tools += [n for n in names if n not in tools]
    return tools


class RuleBackend:
    """Deterministic fallback: no model, no network. Also the reference for expected answers."""

    name = "rules"
    local = True

    def __init__(self) -> None:
        self.last_tools: List[str] = []

    def complete(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Turn:
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        results = {m["name"]: json.loads(m["content"]) for m in messages[last_user + 1:] if m["role"] == "tool"}
        if results or not tools:
            return Turn(content=compose_answer(question, results))
        wanted = detect_tools(question) or list(self.last_tools)  # follow-up: "how fast is it?"
        if not wanted:
            return Turn(content=HELP_TEXT)
        self.last_tools = wanted
        return Turn(tool_calls=[ToolCall(f"call_{i}", name, {}) for i, name in enumerate(wanted)])


HELP_TEXT = ("I can answer questions about this machine's CPU, RAM (including upgrades), disks and free space, "
             "GPU, battery, network adapters, temperatures, USB devices, and model/BIOS/OS. "
             "Try: \"How much RAM do I have?\" or \"Is my SSD NVMe?\"")


def _gaps(result: Dict[str, Any]) -> List[str]:
    out = []
    for g in result.get("unavailable", []):
        line = f"I couldn't determine {g['what']}: {g['reason']}."
        if g.get("how_to_enable"):
            line += f" To enable it, {g['how_to_enable']}."
        out.append(line)
    return out


def _fmt_gb(x: Optional[float]) -> str:
    if x is None:
        return "?"
    return f"{x:g} GB"


def answer_memory(q: str, r: Dict[str, Any]) -> str:
    d = r["data"]
    up = d.get("upgrade")
    wants_upgrade = re.search(UPGRADE, q.lower()) is not None
    parts = []
    if up and wants_upgrade:
        if up["total_slots"] == 0:
            return f"You have {_fmt_gb(d['installed_gb'])} of RAM and it is all soldered to the motherboard, so it cannot be upgraded."
        parts.append(f"You have {_fmt_gb(d['installed_gb'])} in {up['used_slots']} of {up['total_slots']} slots"
                     + (f" (plus {_fmt_gb(up['soldered_gb'])} soldered)" if up.get("soldered_gb") else "") + ".")
        if up.get("max_supported_gb"):
            if up["free_slots"] > 0:
                parts.append(f"The board supports up to {_fmt_gb(up['max_supported_gb'])}, so you can add a stick of up to "
                             f"{_fmt_gb(up['max_per_slot_gb'])} (for {_fmt_gb(up['max_total_gb_keeping_current_modules'])} total).")
            else:
                parts.append(f"All slots are in use. The board supports up to {_fmt_gb(up['max_supported_gb'])}, so to go "
                             f"higher you would replace modules with ones of up to {_fmt_gb(up['max_per_slot_gb'])} each.")
            if up.get("buy"):
                parts.append(f"Buy {up['buy']}.")
            parts.append("(The maximum comes from the firmware; confirm it in the laptop's service manual.)")
        return " ".join(parts + _gaps(r))
    if d.get("installed_gb"):
        line = f"You have {_fmt_gb(d['installed_gb'])} of RAM installed"
        if d.get("type"):
            line += f" ({d['type']}" + (f", {d['speed_mts']} MT/s" if d.get("speed_mts") else "") + ")"
        parts.append(line + f"; {_fmt_gb(d.get('usable_gb'))} is usable by the OS and {_fmt_gb(d.get('available_now_gb'))} is free right now.")
    elif d.get("usable_gb"):
        parts.append(f"You have {_fmt_gb(d['usable_gb'])} of usable RAM, with {_fmt_gb(d.get('available_now_gb'))} free right now.")
    return " ".join(parts + _gaps(r)) or "I couldn't read memory information."


def answer_storage(q: str, r: Dict[str, Any]) -> str:
    disks = r["data"].get("disks", [])
    if not disks:
        return " ".join(_gaps(r)) or "I couldn't find any disks."
    lines = [f"{d['device']}: {d.get('model') or 'unknown model'}, {d.get('size') or '?'} {d['kind']}" for d in disks]
    ql = q.lower()
    lead = ""
    if "nvme" in ql:
        nvme = [d for d in disks if d["kind"] == "NVMe SSD"]
        lead = f"Yes, {nvme[0]['device']} is an NVMe SSD. " if nvme else "No, none of your disks is NVMe. "
    elif re.search(r"\bssd\b", ql) and re.search(r"\b(hdd|hard)\b", ql):
        lead = ""
    elif re.search(r"\b(ssd|hdd|hard drive)\b", ql):
        ssd = [d for d in disks if "SSD" in d["kind"]]
        lead = (f"You have {len(ssd)} SSD(s). " if ssd else "You have no SSD. ")
    return (lead + "Disks: " + "; ".join(lines) + ". " + " ".join(_gaps(r))).strip()


def answer_disk_usage(q: str, r: Dict[str, Any]) -> str:
    fs = r["data"].get("filesystems", [])
    if not fs:
        return " ".join(_gaps(r)) or "No mounted filesystems found."
    return "Free space: " + "; ".join(f"{f['mountpoint']} has {f['free']} free of {f['size']} ({f['used_percent']}% used)"
                                      for f in fs) + "."


def answer_cpu(q: str, r: Dict[str, Any]) -> str:
    d = r["data"]
    if not d.get("model"):
        return " ".join(_gaps(r)) or "I couldn't read CPU information."
    s = f"Your CPU is an {d['model']}" if d["model"][0].lower() in "aeiou" else f"Your CPU is a {d['model']}"
    if d.get("cores"):
        s += f" with {d['cores']} cores and {d.get('threads')} threads"
    elif d.get("threads"):
        s += f" with {d['threads']} threads"
    if d.get("max_mhz"):
        s += f", boosting up to {d['max_mhz'] / 1000:.2f} GHz"
    s += "."
    isa = d.get("instruction_sets", {})
    if re.search(r"avx|instruction", q.lower()):
        s += " AVX2: " + ("yes" if isa.get("avx2") else "no") + ", AVX-512: " + ("yes" if isa.get("avx512f") else "no") + "."
    if re.search(r"virtuali", q.lower()):
        s += f" Hardware virtualization: {d.get('virtualization') or 'not reported'}."
    return " ".join([s] + _gaps(r))


def answer_gpu(q: str, r: Dict[str, Any]) -> str:
    gpus = r["data"].get("gpus", [])
    if not gpus:
        return " ".join(_gaps(r)) or "I couldn't find a GPU."
    parts = []
    for g in gpus:
        s = f"{g.get('name') or (g.get('vendor') or '') + ' ' + (g.get('model') or '')}".strip()
        extra = []
        if g.get("vram_gb"):
            extra.append(f"{_fmt_gb(g['vram_gb'])} VRAM")
        if g.get("driver"):
            extra.append(f"driver {g['driver']}")
        parts.append(s + (f" ({', '.join(extra)})" if extra else ""))
    lead = f"You have {len(gpus)} GPU{'s' if len(gpus) > 1 else ''}: "
    return " ".join([lead + "; ".join(parts) + "."] + _gaps(r))


def answer_battery(q: str, r: Dict[str, Any]) -> str:
    bats = r["data"].get("batteries", [])
    if not bats:
        return " ".join(_gaps(r))
    out = []
    for b in bats:
        s = f"Battery {b['name']} is at {b.get('charge_percent')}% ({(b.get('status') or 'unknown').lower()})."
        if b.get("health_percent") is not None:
            unit = "wh" if "full_capacity_wh" in b else "ah"
            s += (f" Health is {b['health_percent']}% of its design capacity ({b[f'full_capacity_{unit}']} of "
                  f"{b[f'design_capacity_{unit}']} {unit.capitalize()}).")
        if b.get("cycle_count"):
            s += f" It has gone through {b['cycle_count']} charge cycles."
        out.append(s)
    return " ".join(out + [g for g in _gaps(r) if "cycle" not in g or not any(b.get("cycle_count") for b in bats)])


def answer_network(q: str, r: Dict[str, Any]) -> str:
    d = r["data"]
    adapters = d.get("adapters", [])
    parts = [f"{a['interface']} ({a['kind']}, {a.get('state') or 'unknown'}"
             + (f", {a['link_speed_mbps']} Mbps" if a.get("link_speed_mbps") else "")
             + (f", driver {a['driver']}" if a.get("driver") else "") + ")" for a in adapters]
    s = ("Network adapters: " + "; ".join(parts) + ".") if parts else "No physical network adapters found."
    if d.get("controllers"):
        s += " Controllers: " + "; ".join(d["controllers"]) + "."
    return s


def answer_temperatures(q: str, r: Dict[str, Any]) -> str:
    sensors = r["data"].get("sensors", [])
    if not sensors:
        return " ".join(_gaps(r))
    hottest = max(sensors, key=lambda s: s["celsius"])
    return ("Current temperatures: " + "; ".join(f"{s['sensor']} {s['celsius']}°C" for s in sensors)
            + f". The hottest sensor is {hottest['sensor']} at {hottest['celsius']}°C.")


def answer_usb(q: str, r: Dict[str, Any]) -> str:
    devs = r["data"].get("devices")
    if devs is None:
        return " ".join(_gaps(r))
    return "USB devices: " + ("; ".join(d["name"] or f"{d['vendor_id']}:{d['product_id']}" for d in devs) or "none") + "."


def answer_system(q: str, r: Dict[str, Any]) -> str:
    d = r["data"]
    parts = []
    if d.get("manufacturer") or d.get("model"):
        parts.append(f"This is a {d.get('manufacturer') or ''} {d.get('model') or ''}".strip()
                     + (f" ({d['form_factor']})" if d.get("form_factor") else "") + ".")
    if d.get("os"):
        parts.append(f"It runs {d['os']}" + (f" with Linux kernel {d['kernel']}" if d.get("kernel") else "") + ".")
    bios = d.get("bios") or {}
    if bios.get("version"):
        parts.append(f"BIOS {bios['version']}" + (f" ({bios['date']})" if bios.get("date") else "") + ".")
    return " ".join(parts + _gaps(r))


ANSWERERS = {
    "get_system_info": answer_system, "get_cpu_info": answer_cpu, "get_memory_info": answer_memory,
    "get_storage_info": answer_storage, "get_disk_usage": answer_disk_usage, "get_gpu_info": answer_gpu,
    "get_battery_info": answer_battery, "get_network_adapters": answer_network,
    "get_temperatures": answer_temperatures, "get_usb_devices": answer_usb,
}


def compose_answer(question: str, results: Dict[str, Dict[str, Any]]) -> str:
    if not results:
        return HELP_TEXT
    parts = []
    for name in ANSWERERS:
        if name in results:
            if "error" in results[name]:
                parts.append(f"({name} failed: {results[name]['error']})")
            else:
                parts.append(ANSWERERS[name](question, results[name]))
    return "\n".join(p for p in parts if p)
