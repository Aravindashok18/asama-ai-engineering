"""Post-answer check: every number-with-a-unit in the answer must be backed by tool data.

LLMs are good at phrasing and bad at not inventing specs ("supports up to 32 GB" from
general knowledge of the model line). This check is cheap and catches exactly that class
of error; the assistant then retries once with the offending numbers named, and if the
retry still fails it tells the user which figure it could not verify.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, List, Set

UNIT_RE = re.compile(
    r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(tb|gb|gib|mb|mib|ghz|mhz|mt/s|w|wh|mah|%|°c|c\b|cores?|threads?|slots?|mbps|gbps)(?![\w])",
    re.I,
)


def _numbers_in(obj: Any, out: Set[float]) -> None:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        out.add(float(obj))
        if obj > 10**6:  # raw byte counts -> the forms an answer would use
            for div in (10**9, 10**12, 1024**3, 1024**4):
                out.add(round(obj / div, 2))
        if obj >= 1000:  # MHz -> GHz
            out.add(round(obj / 1000, 2))
        return
    if isinstance(obj, str):
        # numbers inside text ("16 MiB", "DDR4 3200") count as-is, without unit conversions,
        # so a model number like "5825U" cannot vouch for "5.8 GHz"
        for m in re.finditer(r"\d+(?:\.\d+)?", obj):
            out.add(float(m.group(0)))
        return
    if isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)


def evidence_numbers(tool_results: Iterable[Any]) -> Set[float]:
    out: Set[float] = set()
    for r in tool_results:
        _numbers_in(r, out)
    return out


def _supported(value: float, evidence: Set[float]) -> bool:
    for e in evidence:
        if abs(e - value) <= max(0.02 * abs(e), 0.051):  # allows rounding (4.546 -> 4.5), not invention
            return True
    return False


def ungrounded_claims(answer: str, tool_results: Iterable[Any]) -> List[str]:
    # No arithmetic is allowed in the answer: derived figures (free slots, addable RAM,
    # battery health) are computed by the tools, so they are already in the evidence.
    evidence = evidence_numbers(tool_results)
    return [m.group(0) for m in UNIT_RE.finditer(answer)
            if not _supported(float(m.group(1).replace(",", ".")), evidence)]
