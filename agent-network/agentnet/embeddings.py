"""Text embeddings and an in-memory vector index.

The default HashingEmbedder needs no model download, is deterministic (good for tests)
and is fast, but it only captures word overlap. For real semantic matching switch to
AGENTNET_EMBEDDER=ollama (nomic-embed-text) or plug in any model behind `Embedder`.
"""
from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from typing import Dict, Hashable, Iterable, List, Optional, Protocol, Tuple

import numpy as np

_TOKEN = re.compile(r"[a-z0-9₹%]+")
_STOP = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at", "from",
    "is", "are", "be", "this", "that", "these", "those", "it", "its", "as", "we", "our", "us",
    "you", "your", "all", "any", "will", "can", "into", "about", "via", "per", "up", "so",
    "interested", "interest", "want", "looking", "like", "such", "also", "more", "new",
}


def _stem(word: str) -> str:
    for suffix, repl in (("ies", "y"), ("ings", ""), ("ing", ""), ("ers", "er"), ("ed", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix) and not word.endswith("ss"):
            return word[: -len(suffix)] + repl
    return word


def tokenize(text: str) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Signed feature hashing over unigrams + bigrams with sublinear tf, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        toks = tokenize(text)
        feats = Counter(toks)
        for a, b in zip(toks, toks[1:]):
            feats[a + "_" + b] += 1
        vec = np.zeros(self.dim, dtype=np.float32)
        for feat, count in feats.items():
            h = zlib.crc32(feat.encode("utf-8"))
            sign = 1.0 if (h >> 16) & 1 else -1.0
            weight = (1.0 + math.log(count)) * (0.6 if "_" in feat else 1.0)
            vec[h % self.dim] += sign * weight
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class OllamaEmbedder:
    """Calls a local Ollama server (POST /api/embed). Data never leaves the host."""

    name = "ollama"

    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        import httpx

        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=30.0)

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.post("/api/embed", json={"model": self.model, "input": text})
        resp.raise_for_status()
        vec = np.asarray(resp.json()["embeddings"][0], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec


class FastEmbedEmbedder:
    """BAAI/bge-small-en-v1.5 via ONNX runtime (pip install fastembed). Runs locally on CPU."""

    name = "fastembed"
    _models: Dict[str, object] = {}

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding

        if model not in self._models:  # loading the ONNX model is the slow part; share it
            self._models[model] = TextEmbedding(model)
        self._model = self._models[model]
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text not in self._cache:
            vec = np.asarray(next(iter(self._model.embed([text]))), dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            self._cache[text] = vec / norm if norm else vec
        return self._cache[text]


def make_embedder(kind: str, ollama_url: str = "", ollama_model: str = "") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind == "fastembed":
        return FastEmbedEmbedder()
    if kind == "ollama":
        return OllamaEmbedder(model=ollama_model or "nomic-embed-text", base_url=ollama_url or "http://localhost:11434")
    raise ValueError(f"unknown embedder {kind!r}")


def to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class VectorIndex:
    """Exact cosine search over unit vectors, grouped by owner.

    Stand-in for pgvector HNSW / FAISS: same interface (upsert, delete, top-k), exact
    results. Exact search over 10k agents x ~5 interests is ~50k dot products, which is
    still only a few milliseconds with numpy; past ~1M rows swap in an ANN index.
    """

    def __init__(self) -> None:
        self._keys: List[Tuple[str, Hashable]] = []  # (owner_id, item_id)
        self._vecs: List[np.ndarray] = []
        self._matrix: Optional[np.ndarray] = None
        self._by_owner: Dict[str, List[int]] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def set_owner(self, owner: str, items: Iterable[Tuple[Hashable, np.ndarray]]) -> None:
        """Replace every vector of one owner (e.g. an agent's interest statements)."""
        self.remove_owner(owner)
        for item_id, vec in items:
            self._keys.append((owner, item_id))
            self._vecs.append(np.asarray(vec, dtype=np.float32))
        self._reindex()

    def remove_owner(self, owner: str) -> None:
        if owner not in self._by_owner:
            return
        keep = [i for i, (o, _) in enumerate(self._keys) if o != owner]
        self._keys = [self._keys[i] for i in keep]
        self._vecs = [self._vecs[i] for i in keep]
        self._reindex()

    def _reindex(self) -> None:
        self._matrix = np.vstack(self._vecs) if self._vecs else None
        self._by_owner = {}
        for i, (owner, _) in enumerate(self._keys):
            self._by_owner.setdefault(owner, []).append(i)

    def search(self, query: np.ndarray, k: int, min_score: float = -1.0) -> List[Tuple[str, float]]:
        """Top-k owners by their best-matching vector."""
        if self._matrix is None:
            return []
        sims = self._matrix @ query
        best: Dict[str, float] = {}
        for idx in np.argsort(-sims):
            score = float(sims[idx])
            if score < min_score:
                break
            owner = self._keys[idx][0]
            if owner not in best:
                best[owner] = score
                if len(best) >= k:
                    break
        return sorted(best.items(), key=lambda kv: -kv[1])

    def max_sim(self, owner: str, query: np.ndarray) -> float:
        rows = self._by_owner.get(owner)
        if not rows or self._matrix is None:
            return 0.0
        return float(np.max(self._matrix[rows] @ query))
