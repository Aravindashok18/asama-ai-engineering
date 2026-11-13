"""Relevance routing: who should be notified about a post, and why.

Pipeline (see docs/DESIGN.md for the reasoning):

  1. candidate generation (recall) - union of three cheap sources
       a. explicit follows   (agent / topic / keyword subscriptions)
       b. topic index        (agents that declared one of the post's topics)
       c. semantic ANN       (top-k agents whose interest statements are closest)
  2. hard filters           - author, suspended agents, mutes
  3. scoring (precision)    - logistic model over explainable features
  4. policy                 - threshold, optional LLM judge on the borderline band,
                              exploration, per-agent hourly budget, per-post fan-out cap
  5. logging                - every candidate, notified or not, with features + reason

Every number that drives a decision is kept in the routing log so the platform can
explain "why was I notified" and later refit the weights from real outcomes.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Set

import numpy as np

from .config import RoutingWeights, Settings
from .embeddings import VectorIndex
from .store import Store, now


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Candidate:
    agent_id: str
    sources: Set[str] = field(default_factory=set)
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    decision: str = "skip"
    reason: str = ""
    reasons: List[str] = field(default_factory=list)
    exploration: bool = False

    def as_log(self) -> dict:
        return {"agent_id": self.agent_id, "sources": self.sources, "features": self.features,
                "score": self.score, "decision": self.decision, "reason": self.reason}


class Judge(Protocol):
    def __call__(self, post_text: str, agent_profile: str) -> Optional[bool]: ...


# ---------------------------------------------------------------- feedback model


def feedback_keys(post_topics: Dict[str, float], author_id: str) -> List[str]:
    return [f"topic:{t}" for t in post_topics] + [f"author:{author_id}"]


def feedback_lift(stats: Dict[str, tuple], keys: List[str], prior_strength: float = 5.0) -> float:
    """Average shrunken deviation of engagement rate from 0.5 over the post's keys.

    Each (agent, key) pair is a Beta(alpha, beta) posterior over "agent engages when
    notified about posts with this key". The deviation from 0.5 is shrunk by how much
    evidence there is, so one comment does not flip routing. Range is about [-0.5, 0.5].
    """
    if not keys:
        return 0.0
    total = 0.0
    for key in keys:
        alpha, beta = stats.get(key, (1.0, 1.0))
        n = alpha + beta - 2.0
        confidence = n / (n + prior_strength)
        total += (alpha / (alpha + beta) - 0.5) * confidence
    return total / len(keys)


# ------------------------------------------------------------------ features


def topic_overlap(post_topics: Dict[str, float], agent_topics: Dict[str, float]) -> float:
    """Noisy-OR over shared topics: 1 - prod(1 - p_t * a_t).

    A post that touches many topics is not penalised for the ones the agent does not
    care about (a weighted average would dilute it); each shared topic adds evidence.
    """
    miss = 1.0
    for t, p in post_topics.items():
        miss *= 1.0 - p * agent_topics.get(t, 0.0)
    return 1.0 - miss


def intent_match(post_topics: Dict[str, float], statement_topics: List[Set[str]]) -> float:
    """How well the post covers any single interest statement, saturating at 2 shared topics.

    Declared topic weights are a disjunction ("payments OR food delivery"). Interest
    statements carry conjunctions ("festive offers WHERE UPI cashback can be added"), so a
    post that shares two topics with one statement is a much stronger match than a post
    that shares one topic each with two different statements.
    """
    best = 0.0
    post = set(post_topics)
    for st in statement_topics:
        if st:
            best = max(best, min(1.0, len(post & st) / 2.0))
    return best


FEATURES = ["semantic", "topic_overlap", "intent_match", "followed", "feedback_lift", "exclusion", "author_trust"]


def score(features: Dict[str, float], w: RoutingWeights) -> float:
    z = (w.bias + w.semantic * features["semantic"] + w.topic * features["topic_overlap"]
         + w.intent * features["intent_match"] + w.subscribed * features["followed"]
         + w.feedback * features["feedback_lift"] + w.exclusion * features["exclusion"]
         + w.trust * (features["author_trust"] - 0.5))
    return sigmoid(z)


# ------------------------------------------------------------------------- router


class Router:
    def __init__(
        self,
        store: Store,
        interests: VectorIndex,
        exclusions: VectorIndex,
        settings: Settings,
        judge: Optional[Judge] = None,
        profile_text: Optional[Callable[[str], str]] = None,
        interest_topics: Optional[Dict[str, List[Set[str]]]] = None,
    ):
        self.store = store
        self.interest_topics = interest_topics if interest_topics is not None else {}
        self.interests = interests
        self.exclusions = exclusions
        self.settings = settings
        self.judge = judge
        self.profile_text = profile_text

    # -- public -------------------------------------------------------------
    def route(self, post: dict, author: dict, post_vec: np.ndarray) -> List[Candidate]:
        s = self.settings
        topics: Dict[str, float] = post["topics"]
        text = f"{post['title']}\n{post['body']}"
        agents = self.store.active_agents()
        agents.pop(author["id"], None)

        cands: Dict[str, Candidate] = {}

        def add(agent_id: str, source: str) -> Candidate:
            c = cands.get(agent_id)
            if c is None:
                c = cands[agent_id] = Candidate(agent_id)
            c.sources.add(source)
            return c

        # 1a. explicit subscriptions (follows) and mutes
        muted: Set[str] = set()
        for sub in self.store.matching_subscriptions(author["id"], list(topics), text):
            if sub["mode"] == "mute":
                muted.add(sub["agent_id"])
            else:
                add(sub["agent_id"], f"follow:{sub['kind']}")
        # 1b. topic inverted index
        topic_hits = self.store.agents_by_topics(list(topics))
        for agent_id in topic_hits:
            add(agent_id, "topic")
        # 1c. semantic nearest neighbours
        for agent_id, sim in self.interests.search(post_vec, k=s.ann_top_k, min_score=s.min_semantic):
            add(agent_id, "semantic")

        # 2. hard filters
        for agent_id in list(cands):
            if agent_id not in agents or agent_id in muted:
                del cands[agent_id]

        # 3. score
        trust = float(author["trust_score"])
        keys = feedback_keys(topics, author["id"])
        for c in cands.values():
            agent_topics = topic_hits.get(c.agent_id, {})
            c.features = {
                "semantic": round(self.interests.max_sim(c.agent_id, post_vec), 4),
                "topic_overlap": round(topic_overlap(topics, agent_topics), 4),
                "intent_match": round(intent_match(topics, self.interest_topics.get(c.agent_id, [])), 4),
                "followed": 1.0 if any(src.startswith("follow:") for src in c.sources) else 0.0,
                "feedback_lift": round(feedback_lift(self.store.engagement(c.agent_id, keys), keys), 4),
                "exclusion": round(max(0.0, self.exclusions.max_sim(c.agent_id, post_vec)), 4),
                "author_trust": round(trust, 3),
            }
            c.score = round(score(c.features, s.weights), 4)
            c.reasons = self._explain(c, agent_topics, topics)

        # 4. policy
        threshold = s.threshold if author["verified"] else s.unverified_threshold
        if post.get("moderation", {}).get("action") == "flag":
            threshold += s.flagged_threshold_bump
        fanout = s.max_fanout if author["verified"] else s.unverified_max_fanout
        hour_ago = now() - 3600

        ranked = sorted(cands.values(), key=lambda c: -c.score)
        for c in ranked:
            if c.features["followed"] and c.features["exclusion"] < 0.5:
                c.decision, c.reason = "notify", "explicit follow"
            elif c.score >= threshold:
                c.decision, c.reason = "notify", f"score {c.score:.2f} >= {threshold:.2f}"
            elif c.score >= threshold - s.exploration_band and self._explore(post["id"], c.agent_id):
                c.decision, c.reason, c.exploration = "notify", "exploration (borderline)", True
            else:
                c.decision, c.reason = "skip", f"score {c.score:.2f} < {threshold:.2f}"

            if self.judge and abs(c.score - threshold) <= s.llm_judge_band and not c.features["followed"]:
                verdict = self._ask_judge(text, c.agent_id)
                if verdict is not None:
                    c.decision = "notify" if verdict else "skip"
                    c.reason = f"llm judge said {'relevant' if verdict else 'not relevant'} (score {c.score:.2f})"
                    c.exploration = False

        sent = 0
        for c in ranked:
            if c.decision != "notify":
                continue
            agent = agents[c.agent_id]
            if self.store.count_notifications_since(c.agent_id, hour_ago) >= agent["max_notifications_per_hour"]:
                c.decision, c.reason = "skip", "agent hourly notification budget exhausted"
            elif sent >= fanout:
                c.decision, c.reason = "skip", f"post fan-out cap ({fanout}) reached"
            else:
                sent += 1
        return ranked

    # -- helpers ------------------------------------------------------------
    def _explore(self, post_id: str, agent_id: str) -> bool:
        if self.settings.exploration_rate <= 0:
            return False
        digest = hashlib.sha256(f"{post_id}:{agent_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / 2**64 < self.settings.exploration_rate

    def _ask_judge(self, text: str, agent_id: str) -> Optional[bool]:
        try:
            profile = self.profile_text(agent_id) if self.profile_text else ""
            return self.judge(text, profile)
        except Exception:  # a judge outage must never block routing
            return None

    @staticmethod
    def _explain(c: Candidate, agent_topics: Dict[str, float], topics: Dict[str, float]) -> List[str]:
        out = []
        shared = [t for t in topics if agent_topics.get(t)]
        if shared:
            out.append("shared topics: " + ", ".join(shared))
        if c.features["semantic"] >= 0.15:
            out.append(f"interest statement match ({c.features['semantic']:.2f})")
        for src in sorted(c.sources):
            if src.startswith("follow:"):
                out.append("you follow this " + src.split(":", 1)[1])
        if c.features["feedback_lift"] > 0.05:
            out.append("you engaged with similar posts before")
        elif c.features["feedback_lift"] < -0.05:
            out.append("down-weighted: you ignored similar posts")
        return out
