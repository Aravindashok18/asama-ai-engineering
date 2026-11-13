"""Measure routing quality on hand-labelled posts.

    python scripts/eval_routing.py                 # dev + test sets, summary
    python scripts/eval_routing.py --set test -v   # per-post detail

Each eval post lists the agents that would plausibly act on it. The router's notified
set is compared against that list (micro precision / recall / F1). Three simple
baselines run over the same candidates to show what each signal adds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentnet.embeddings import make_embedder  # noqa: E402
from agentnet.models import PostCreate  # noqa: E402
from agentnet.seed import DATA_DIR, in_memory_platform, seed_platform  # noqa: E402


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def evaluate(set_name: str, embedder, verbose: bool = False) -> dict:
    posts = json.loads((DATA_DIR / f"eval_{set_name}.json").read_text(encoding="utf-8"))
    systems = {"broadcast": [0, 0, 0], "any_shared_topic": [0, 0, 0], "semantic_top2": [0, 0, 0], "hybrid_router": [0, 0, 0]}
    for item in posts:
        platform = in_memory_platform(embedder)  # fresh platform per post: no rate limits or feedback leakage
        agents = seed_platform(platform)
        handles = {v["agent"]["id"]: h for h, v in agents.items()}
        author = agents[item["author"]]["agent"]
        post = platform.create_post(author, PostCreate(kind=item["kind"], title=item["title"], body=item["body"]))
        ranked = platform.route_post(post["id"])
        truth = set(item["relevant"])
        everyone = {h for h in agents if h != item["author"]}
        predictions = {
            "broadcast": everyone,
            "any_shared_topic": {handles[c.agent_id] for c in ranked if c.features["topic_overlap"] > 0},
            "semantic_top2": {handles[c.agent_id] for c in sorted(ranked, key=lambda c: -c.features["semantic"])[:2]},
            "hybrid_router": {handles[c.agent_id] for c in ranked if c.decision == "notify"},
        }
        for name, pred in predictions.items():
            systems[name][0] += len(pred & truth)
            systems[name][1] += len(pred - truth)
            systems[name][2] += len(truth - pred)
        if verbose:
            pred = predictions["hybrid_router"]
            mark = "ok " if pred == truth else "ERR"
            print(f"[{mark}] {item['author']:>11}: {item['title']}")
            print(f"       topics={post['topics']}")
            print(f"       expected={sorted(truth)} got={sorted(pred)}")
            for c in ranked[:6]:
                f = c.features
                print(f"         {handles[c.agent_id]:<12} score={c.score:.2f} sem={f['semantic']:.2f} "
                      f"topic={f['topic_overlap']:.2f} excl={f['exclusion']:.2f} -> {c.decision}")
    return {name: prf(*v) for name, v in systems.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "test", "both"], default="both")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "fastembed", "ollama"])
    args = ap.parse_args()
    sets = ["dev", "test"] if args.set == "both" else [args.set]
    embedder = make_embedder(args.embedder)
    results = {s: evaluate(s, embedder, args.verbose) for s in sets}
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for s, res in results.items():
        print(f"\n== {s} set ({args.embedder} embedder) ==")
        print(f"{'system':<18}{'precision':>10}{'recall':>8}{'f1':>7}{'tp':>5}{'fp':>5}{'fn':>5}")
        for name, m in res.items():
            print(f"{name:<18}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>7.3f}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}")


if __name__ == "__main__":
    main()
