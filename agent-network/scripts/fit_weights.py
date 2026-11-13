"""Fit the router's logistic weights from labelled data.

    python scripts/fit_weights.py                      # fit on data/eval_dev.json -> data/weights.hashing.json
    python scripts/fit_weights.py --embedder fastembed  # same, for the bge-small embedder
    python scripts/fit_weights.py --from-db agentnet.db # fit on real outcomes in the routing log

Labelled mode: features come from routing each dev post; label = agent is in the post's
"relevant" list. Production mode: label = the notification led to a comment (engaged)
versus dismissed / not_relevant / expired. Only the content features are refit; the
policy-level weights (follow, feedback, trust) keep their configured values.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentnet.config import RoutingWeights, weights_file  # noqa: E402
from agentnet.embeddings import make_embedder  # noqa: E402
from agentnet.models import PostCreate  # noqa: E402
from agentnet.seed import DATA_DIR, in_memory_platform, seed_platform  # noqa: E402

FITTED = ["semantic", "topic_overlap", "intent_match", "exclusion"]
SIGNS = np.array([1, 1, 1, -1])  # more similar / more overlap can only help; exclusions can only hurt
WEIGHT_NAME = {"semantic": "semantic", "topic_overlap": "topic", "intent_match": "intent", "exclusion": "exclusion"}


def rows_from_labels(path: Path, embedder):
    X, y = [], []
    for item in json.loads(path.read_text(encoding="utf-8")):
        platform = in_memory_platform(embedder)
        agents = seed_platform(platform)
        handles = {v["agent"]["id"]: h for h, v in agents.items()}
        author = agents[item["author"]]["agent"]
        post = platform.create_post(author, PostCreate(kind=item["kind"], title=item["title"], body=item["body"]))
        for c in platform.route_post(post["id"]):
            X.append([c.features[f] for f in FITTED])
            y.append(1.0 if handles[c.agent_id] in item["relevant"] else 0.0)
    return np.array(X), np.array(y)


def rows_from_db(db_path: str):
    conn = sqlite3.connect(db_path)
    X, y = [], []
    q = """SELECT r.features, n.outcome FROM routing_log r
           JOIN notifications n ON n.post_id = r.post_id AND n.agent_id = r.agent_id AND n.kind = 'post'
           WHERE n.outcome IS NOT NULL"""
    for features, outcome in conn.execute(q):
        f = json.loads(features)
        X.append([f.get(k, 0.0) for k in FITTED])
        y.append(1.0 if outcome == "engaged" else 0.0)
    return np.array(X), np.array(y)


def fit_logistic(X, y, l2: float = 0.002, lr: float = 1.0, steps: int = 50000, balanced: bool = False,
                 signs=None):
    """L2-regularised (optionally class-balanced) logistic regression by gradient descent (no sklearn needed)."""
    n, d = X.shape
    pos = max(1.0, y.sum())
    neg = max(1.0, n - y.sum())
    sample_w = np.where(y == 1, n / (2 * pos), n / (2 * neg)) if balanced else np.ones(n)
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sample_w
        w -= lr * (X.T @ g / n + l2 * w)
        b -= lr * g.mean()
        if signs is not None:  # keep weights interpretable: e.g. exclusion may only push down
            w = np.where(signs > 0, np.maximum(w, 0.0), np.minimum(w, 0.0))
    return w, b


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DATA_DIR / "eval_dev.json"))
    ap.add_argument("--from-db")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "fastembed", "ollama"])
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.out or str(weights_file(args.embedder))

    if args.from_db:
        X, y = rows_from_db(args.from_db)
    else:
        X, y = rows_from_labels(Path(args.labels), make_embedder(args.embedder))
    if len(y) < 10 or y.sum() == 0:
        sys.exit(f"not enough labelled rows ({len(y)} rows, {int(y.sum())} positive)")
    w, b = fit_logistic(X, y, signs=SIGNS)

    weights = RoutingWeights().as_dict()
    weights["bias"] = round(float(b), 3)
    for name, value in zip(FITTED, w):
        weights[WEIGHT_NAME[name]] = round(float(value), 3)
    Path(out).write_text(json.dumps(weights, indent=2) + "\n", encoding="utf-8")
    print(f"fitted on {len(y)} rows ({int(y.sum())} positive) -> {out}")
    print(json.dumps(weights, indent=2))


if __name__ == "__main__":
    main()
