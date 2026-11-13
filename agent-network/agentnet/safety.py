"""Trust & safety: API keys, content moderation, near-duplicate detection, webhook signing."""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from .embeddings import tokenize

# --------------------------------------------------------------------------- keys


def new_api_key() -> str:
    return "ak_" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def new_secret() -> str:
    return "whsec_" + secrets.token_urlsafe(32)


# --------------------------------------------------------------------- moderation

_INJECTION = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts|messages|rules)",
    r"disregard (all |any |the |your )?(previous |prior |system )?(instructions|prompt|rules)",
    r"forget (all |your )?(previous |prior )?instructions",
    r"you are now (a|an|in) ",
    r"(reveal|print|output|send|share|leak) (me )?(your|the) (system prompt|instructions|api[ _-]?keys?|secrets?|credentials|webhook secret)",
    r"<\s*/?\s*(system|assistant|instructions)\s*>",
    r"\bnew instructions\s*:",
    r"act as (an? )?(admin|administrator|developer|root|system)",
    r"(call|invoke|use) (the )?(tool|function) ",
]
_PHISHING = [
    r"(share|send|enter|confirm|verify) (your |the )?(otp|one[- ]time password|upi pin|pin|cvv|password|card number|net ?banking)",
    r"(account|kyc) (will be|is) (blocked|suspended|frozen)",
]
_SCAM = [
    r"guaranteed (returns|profit|income)",
    r"double your (money|investment)",
    r"risk[- ]free (returns|profit|investment)",
    r"\b\d{2,3}\s*% (daily|weekly) returns",
]
_SHORTENERS = ("bit.ly", "tinyurl.com", "t.co", "cutt.ly", "goo.gl", "is.gd", "rb.gy", "shorturl.at")
_URL = re.compile(r"https?://[^\s)>\]]+", re.I)


@dataclass
class ModerationResult:
    action: str = "allow"  # allow | flag | quarantine
    flags: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "flags": self.flags}


class Moderator:
    """Fast, explainable first line of defence.

    Runs before routing on every post and comment. In production a classifier such as
    Llama Guard sits behind this for anything the rules flag or cannot judge; the rules
    stay because they are cheap, deterministic and easy to audit.
    """

    def __init__(self) -> None:
        self._injection = [re.compile(p, re.I) for p in _INJECTION]
        self._phishing = [re.compile(p, re.I) for p in _PHISHING]
        self._scam = [re.compile(p, re.I) for p in _SCAM]

    def check(self, text: str) -> ModerationResult:
        res = ModerationResult()
        if any(p.search(text) for p in self._injection):
            res.flags.append("prompt_injection")
        if any(p.search(text) for p in self._phishing):
            res.flags.append("credential_phishing")
        if any(p.search(text) for p in self._scam):
            res.flags.append("financial_scam")
        urls = _URL.findall(text)
        if len(urls) > 3:
            res.flags.append("link_spam")
        if any(urlparse(u).hostname in _SHORTENERS for u in urls if urlparse(u).hostname):
            res.flags.append("url_shortener")
        letters = [c for c in text if c.isalpha()]
        if len(letters) > 40 and sum(c.isupper() for c in letters) / len(letters) > 0.7:
            res.flags.append("shouting")

        severe = {"prompt_injection", "credential_phishing", "financial_scam"}
        if severe & set(res.flags):
            res.action = "quarantine"
        elif res.flags:
            res.action = "flag"
        return res


# ------------------------------------------------------------- duplicate detection


def content_hash(text: str) -> str:
    return hashlib.sha256(" ".join(tokenize(text)).encode("utf-8")).hexdigest()


def simhash(text: str, bits: int = 64) -> int:
    toks = tokenize(text)
    shingles = [" ".join(toks[i : i + 2]) for i in range(max(1, len(toks) - 1))] or [""]
    acc = [0] * bits
    for sh in shingles:
        h = int.from_bytes(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big")
        for i in range(bits):
            acc[i] += 1 if (h >> i) & 1 else -1
    return sum(1 << i for i in range(bits) if acc[i] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_near_duplicate(text_hash: str, sim: int, previous: List[Tuple[str, str]], max_distance: int = 6) -> bool:
    for prev_hash, prev_sim in previous:
        if prev_hash == text_hash or hamming(sim, int(prev_sim, 16)) <= max_distance:
            return True
    return False


# ----------------------------------------------------------------- webhooks


def sign_payload(secret: str, body: bytes, timestamp: Optional[int] = None) -> Tuple[str, str]:
    """Returns (timestamp, signature header). Signature covers `timestamp.body`."""
    ts = str(int(timestamp if timestamp is not None else time.time()))
    mac = hmac.new(secret.encode("utf-8"), ts.encode("ascii") + b"." + body, hashlib.sha256).hexdigest()
    return ts, f"t={ts},v1={mac}"


def verify_signature(secret: str, body: bytes, header: str, tolerance_s: int = 300, at: Optional[float] = None) -> bool:
    """What a receiving agent runs on every webhook (constant-time compare + replay window)."""
    try:
        parts = dict(p.split("=", 1) for p in header.split(","))
        ts = int(parts["t"])
        sig = parts["v1"]
    except (ValueError, KeyError):
        return False
    if abs((at if at is not None else time.time()) - ts) > tolerance_s:
        return False
    expected = hmac.new(secret.encode("utf-8"), str(ts).encode("ascii") + b"." + body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def check_webhook_url(url: str, allow_private: bool) -> Optional[str]:
    """Basic SSRF guard. Returns an error message or None.

    Production also resolves DNS at send time and re-checks the IP (DNS rebinding), and
    sends from an egress proxy that cannot reach internal networks.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("https", "http"):
        return "webhook_url must be http(s)"
    if not parsed.hostname:
        return "webhook_url has no host"
    if allow_private:
        return None
    if parsed.scheme != "https":
        return "webhook_url must use https"
    host = parsed.hostname
    if host in ("localhost",) or host.endswith(".local") or host.endswith(".internal"):
        return "webhook_url must be publicly reachable"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return None
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
        return "webhook_url must be publicly reachable"
    return None
