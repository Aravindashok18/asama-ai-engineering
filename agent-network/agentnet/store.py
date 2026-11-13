"""SQLite persistence. The schema mirrors the Postgres + pgvector design in docs/DESIGN.md.

SQLite keeps the project runnable with zero setup; every query here is plain SQL that
ports to Postgres directly (vectors live in pgvector there instead of BLOB columns).
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    id                          TEXT PRIMARY KEY,
    handle                      TEXT NOT NULL UNIQUE,
    name                        TEXT NOT NULL,
    description                 TEXT NOT NULL DEFAULT '',
    industry                    TEXT NOT NULL DEFAULT '',
    industry_topics             TEXT NOT NULL DEFAULT '[]',
    domain                      TEXT,
    delivery_mode               TEXT NOT NULL CHECK (delivery_mode IN ('webhook', 'inbox')),
    webhook_url                 TEXT,
    webhook_secret              TEXT NOT NULL,
    api_key_hash                TEXT NOT NULL UNIQUE,
    verified                    INTEGER NOT NULL DEFAULT 0,
    verification_token          TEXT,
    trust_score                 REAL NOT NULL DEFAULT 0.5,
    status                      TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
    max_notifications_per_hour  INTEGER NOT NULL DEFAULT 30,
    consecutive_failures        INTEGER NOT NULL DEFAULT 0,
    webhook_paused_until        REAL NOT NULL DEFAULT 0,
    created_at                  REAL NOT NULL,
    updated_at                  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_interests (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('interest', 'exclusion')),
    text        TEXT NOT NULL,
    embedding   BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_interests_agent ON agent_interests(agent_id);

CREATE TABLE IF NOT EXISTS agent_topics (
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    topic       TEXT NOT NULL,
    weight      REAL NOT NULL,
    PRIMARY KEY (agent_id, topic)
);
CREATE INDEX IF NOT EXISTS idx_agent_topics_topic ON agent_topics(topic);

CREATE TABLE IF NOT EXISTS subscriptions (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL CHECK (kind IN ('topic', 'agent', 'keyword')),
    value       TEXT NOT NULL,
    mode        TEXT NOT NULL CHECK (mode IN ('follow', 'mute')),
    created_at  REAL NOT NULL,
    UNIQUE (agent_id, kind, value)
);
CREATE INDEX IF NOT EXISTS idx_subs_lookup ON subscriptions(kind, value);

CREATE TABLE IF NOT EXISTS posts (
    id               TEXT PRIMARY KEY,
    author_id        TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    kind             TEXT NOT NULL,
    title            TEXT NOT NULL,
    body             TEXT NOT NULL,
    topics           TEXT NOT NULL DEFAULT '{}',
    declared_topics  TEXT NOT NULL DEFAULT '[]',
    region           TEXT,
    content_hash     TEXT NOT NULL,
    simhash          TEXT NOT NULL,
    status           TEXT NOT NULL CHECK (status IN ('published', 'quarantined', 'removed')),
    moderation       TEXT NOT NULL DEFAULT '{}',
    routed_at        REAL,
    created_at       REAL NOT NULL,
    expires_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_posts_author_created ON posts(author_id, created_at);

CREATE TABLE IF NOT EXISTS comments (
    id          TEXT PRIMARY KEY,
    post_id     TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    author_id   TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'visible',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id, created_at);
CREATE INDEX IF NOT EXISTS idx_comments_author ON comments(author_id, created_at);

CREATE TABLE IF NOT EXISTS notifications (
    id               TEXT PRIMARY KEY,
    kind             TEXT NOT NULL CHECK (kind IN ('post', 'comment')),
    post_id          TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    comment_id       TEXT REFERENCES comments(id) ON DELETE CASCADE,
    agent_id         TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    score            REAL,
    reasons          TEXT NOT NULL DEFAULT '[]',
    exploration      INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL CHECK (status IN ('inbox', 'queued', 'delivered', 'failed')),
    outcome          TEXT CHECK (outcome IN ('engaged', 'dismissed', 'not_relevant', 'expired')),
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  REAL,
    last_error       TEXT,
    created_at       REAL NOT NULL,
    delivered_at     REAL,
    outcome_at       REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_post_notification ON notifications(post_id, agent_id) WHERE kind = 'post';
CREATE INDEX IF NOT EXISTS idx_notif_agent ON notifications(agent_id, created_at);
CREATE INDEX IF NOT EXISTS idx_notif_queue ON notifications(status, next_attempt_at);

CREATE TABLE IF NOT EXISTS routing_log (
    post_id     TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    sources     TEXT NOT NULL,
    features    TEXT NOT NULL,
    score       REAL NOT NULL,
    decision    TEXT NOT NULL,
    reason      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (post_id, agent_id)
);

CREATE TABLE IF NOT EXISTS engagement_stats (
    agent_id    TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    key         TEXT NOT NULL,
    alpha       REAL NOT NULL DEFAULT 1.0,
    beta        REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (agent_id, key)
);

CREATE TABLE IF NOT EXISTS reports (
    id           TEXT PRIMARY KEY,
    post_id      TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    reporter_id  TEXT NOT NULL REFERENCES agents(id) ON DELETE CASCADE,
    reason       TEXT NOT NULL,
    created_at   REAL NOT NULL,
    UNIQUE (post_id, reporter_id)
);
"""

JSON_COLUMNS = {"industry_topics", "topics", "declared_topics", "moderation", "reasons", "sources", "features"}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"


def now() -> float:
    return time.time()


def _row(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    out = dict(row)
    for col in JSON_COLUMNS & out.keys():
        out[col] = json.loads(out[col]) if out[col] is not None else None
    return out


class Store:
    def __init__(self, path: str = ":memory:"):
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self._conn.executescript(SCHEMA)

    # ------------------------------------------------------------------ helpers
    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        with self._lock:
            return _row(self._conn.execute(sql, params).fetchone())

    def all(self, sql: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._lock:
            return [_row(r) for r in self._conn.execute(sql, params).fetchall()]

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return row[0] if row else None

    def transaction(self):
        store = self

        class _Tx:
            def __enter__(self_inner):
                store._lock.acquire()
                store._conn.execute("BEGIN IMMEDIATE")
                return store

            def __exit__(self_inner, exc_type, exc, tb):
                try:
                    store._conn.execute("ROLLBACK" if exc_type else "COMMIT")
                finally:
                    store._lock.release()
                return False

        return _Tx()

    @staticmethod
    def _placeholders(n: int) -> str:
        return ",".join("?" * n)

    # ------------------------------------------------------------------ agents
    def insert_agent(self, agent: Dict[str, Any]) -> None:
        cols = list(agent)
        vals = [json.dumps(agent[c]) if c in JSON_COLUMNS else agent[c] for c in cols]
        self.execute(f"INSERT INTO agents ({','.join(cols)}) VALUES ({self._placeholders(len(cols))})", vals)

    def update_agent(self, agent_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now()
        sets = ", ".join(f"{k} = ?" for k in fields)
        vals = [json.dumps(v) if k in JSON_COLUMNS else v for k, v in fields.items()]
        self.execute(f"UPDATE agents SET {sets} WHERE id = ?", [*vals, agent_id])

    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM agents WHERE id = ?", (agent_id,))

    def get_agent_by_handle(self, handle: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM agents WHERE handle = ?", (handle,))

    def get_agent_by_key_hash(self, key_hash: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM agents WHERE api_key_hash = ?", (key_hash,))

    def active_agents(self) -> Dict[str, Dict[str, Any]]:
        return {a["id"]: a for a in self.all("SELECT * FROM agents WHERE status = 'active'")}

    def replace_interests(self, agent_id: str, rows: Iterable[Dict[str, Any]]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM agent_interests WHERE agent_id = ?", (agent_id,))
            for r in rows:
                self._conn.execute(
                    "INSERT INTO agent_interests (id, agent_id, kind, text, embedding) VALUES (?,?,?,?,?)",
                    (r["id"], agent_id, r["kind"], r["text"], r["embedding"]),
                )

    def interests(self, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if agent_id:
            return self.all("SELECT * FROM agent_interests WHERE agent_id = ?", (agent_id,))
        return self.all("SELECT * FROM agent_interests")

    def replace_topics(self, agent_id: str, topics: Dict[str, float]) -> None:
        with self.transaction():
            self._conn.execute("DELETE FROM agent_topics WHERE agent_id = ?", (agent_id,))
            for topic, weight in topics.items():
                self._conn.execute("INSERT INTO agent_topics VALUES (?,?,?)", (agent_id, topic, weight))

    def agent_topics(self, agent_id: str) -> Dict[str, float]:
        return {r["topic"]: r["weight"] for r in self.all("SELECT topic, weight FROM agent_topics WHERE agent_id = ?", (agent_id,))}

    def agents_by_topics(self, topics: Sequence[str]) -> Dict[str, Dict[str, float]]:
        """Inverted index lookup: agent_id -> {topic: weight} for the given topics."""
        if not topics:
            return {}
        rows = self.all(
            f"SELECT agent_id, topic, weight FROM agent_topics WHERE topic IN ({self._placeholders(len(topics))})",
            list(topics),
        )
        out: Dict[str, Dict[str, float]] = {}
        for r in rows:
            out.setdefault(r["agent_id"], {})[r["topic"]] = r["weight"]
        return out

    # ------------------------------------------------------------ subscriptions
    def add_subscription(self, agent_id: str, kind: str, value: str, mode: str) -> Dict[str, Any]:
        existing = self.one("SELECT * FROM subscriptions WHERE agent_id=? AND kind=? AND value=?", (agent_id, kind, value))
        if existing:
            self.execute("UPDATE subscriptions SET mode = ? WHERE id = ?", (mode, existing["id"]))
            existing["mode"] = mode
            return existing
        sub = {"id": new_id("sub"), "agent_id": agent_id, "kind": kind, "value": value, "mode": mode, "created_at": now()}
        self.execute("INSERT INTO subscriptions VALUES (:id,:agent_id,:kind,:value,:mode,:created_at)", sub)
        return sub

    def list_subscriptions(self, agent_id: str) -> List[Dict[str, Any]]:
        return self.all("SELECT * FROM subscriptions WHERE agent_id = ? ORDER BY created_at", (agent_id,))

    def delete_subscription(self, agent_id: str, sub_id: str) -> bool:
        return self.execute("DELETE FROM subscriptions WHERE id = ? AND agent_id = ?", (sub_id, agent_id)).rowcount > 0

    def matching_subscriptions(self, author_id: str, topics: Sequence[str], text: str) -> List[Dict[str, Any]]:
        params: List[Any] = [author_id, *topics]
        topic_clause = f"OR (kind = 'topic' AND value IN ({self._placeholders(len(topics))}))" if topics else ""
        rows = self.all(
            f"SELECT * FROM subscriptions WHERE (kind = 'agent' AND value = ?) {topic_clause}", params
        )
        lowered = text.lower()
        # Keyword subscriptions are few per agent; at scale this becomes an Aho-Corasick
        # automaton (or a Postgres full-text query) rebuilt when keyword subs change.
        for sub in self.all("SELECT * FROM subscriptions WHERE kind = 'keyword'"):
            pattern = r"(?<![a-z0-9])" + re.escape(sub["value"].lower()) + r"(?![a-z0-9])"
            if re.search(pattern, lowered):
                rows.append(sub)
        return rows

    # ------------------------------------------------------------------- posts
    def insert_post(self, post: Dict[str, Any]) -> None:
        cols = list(post)
        vals = [json.dumps(post[c]) if c in JSON_COLUMNS else post[c] for c in cols]
        self.execute(f"INSERT INTO posts ({','.join(cols)}) VALUES ({self._placeholders(len(cols))})", vals)

    def get_post(self, post_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM posts WHERE id = ?", (post_id,))

    def recent_posts(self, limit: int = 50, author_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if author_id:
            return self.all("SELECT * FROM posts WHERE status='published' AND author_id=? ORDER BY created_at DESC LIMIT ?", (author_id, limit))
        return self.all("SELECT * FROM posts WHERE status='published' ORDER BY created_at DESC LIMIT ?", (limit,))

    def count_posts_since(self, author_id: str, since: float) -> int:
        return self.scalar("SELECT COUNT(*) FROM posts WHERE author_id = ? AND created_at >= ?", (author_id, since))

    def author_posts_since(self, author_id: str, since: float) -> List[Dict[str, Any]]:
        return self.all("SELECT id, content_hash, simhash FROM posts WHERE author_id = ? AND created_at >= ?", (author_id, since))

    # ---------------------------------------------------------------- comments
    def insert_comment(self, comment: Dict[str, Any]) -> None:
        self.execute("INSERT INTO comments VALUES (:id,:post_id,:author_id,:body,:status,:created_at)", comment)

    def comments_for(self, post_id: str) -> List[Dict[str, Any]]:
        return self.all(
            """SELECT c.*, a.handle AS author_handle, a.name AS author_name
               FROM comments c JOIN agents a ON a.id = c.author_id
               WHERE c.post_id = ? AND c.status = 'visible' ORDER BY c.created_at""",
            (post_id,),
        )

    def count_comments_since(self, author_id: str, since: float) -> int:
        return self.scalar("SELECT COUNT(*) FROM comments WHERE author_id = ? AND created_at >= ?", (author_id, since))

    # ----------------------------------------------------------- notifications
    def insert_notification(self, n: Dict[str, Any]) -> None:
        cols = list(n)
        vals = [json.dumps(n[c]) if c in JSON_COLUMNS else n[c] for c in cols]
        self.execute(f"INSERT INTO notifications ({','.join(cols)}) VALUES ({self._placeholders(len(cols))})", vals)

    def get_notification(self, notif_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM notifications WHERE id = ?", (notif_id,))

    def post_notification_for(self, post_id: str, agent_id: str) -> Optional[Dict[str, Any]]:
        return self.one("SELECT * FROM notifications WHERE kind='post' AND post_id=? AND agent_id=?", (post_id, agent_id))

    def notifications_for_agent(self, agent_id: str, pending_only: bool, limit: int, after: float = 0.0) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM notifications WHERE agent_id = ? AND created_at > ?"
        if pending_only:
            sql += " AND outcome IS NULL"
        sql += " ORDER BY created_at LIMIT ?"
        return self.all(sql, (agent_id, after, limit))

    def notifications_for_post(self, post_id: str) -> List[Dict[str, Any]]:
        return self.all("SELECT * FROM notifications WHERE post_id = ? AND kind='post' ORDER BY score DESC", (post_id,))

    def count_notifications_since(self, agent_id: str, since: float) -> int:
        return self.scalar("SELECT COUNT(*) FROM notifications WHERE agent_id=? AND kind='post' AND created_at >= ?", (agent_id, since))

    def due_webhooks(self, at: float, limit: int) -> List[Dict[str, Any]]:
        return self.all(
            "SELECT * FROM notifications WHERE status = 'queued' AND next_attempt_at <= ? ORDER BY next_attempt_at LIMIT ?",
            (at, limit),
        )

    def update_notification(self, notif_id: str, **fields: Any) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        self.execute(f"UPDATE notifications SET {sets} WHERE id = ?", [*fields.values(), notif_id])

    def stale_notifications(self, before: float) -> List[Dict[str, Any]]:
        return self.all("SELECT * FROM notifications WHERE kind='post' AND outcome IS NULL AND created_at < ?", (before,))

    # ------------------------------------------------------------- routing log
    def write_routing_log(self, post_id: str, rows: Iterable[Dict[str, Any]]) -> None:
        ts = now()
        with self.transaction():
            for r in rows:
                self._conn.execute(
                    "INSERT OR REPLACE INTO routing_log VALUES (?,?,?,?,?,?,?,?)",
                    (post_id, r["agent_id"], json.dumps(sorted(r["sources"])), json.dumps(r["features"]),
                     r["score"], r["decision"], r["reason"], ts),
                )

    def routing_log(self, post_id: str) -> List[Dict[str, Any]]:
        return self.all(
            """SELECT r.*, a.handle FROM routing_log r JOIN agents a ON a.id = r.agent_id
               WHERE r.post_id = ? ORDER BY r.score DESC""",
            (post_id,),
        )

    # ---------------------------------------------------------------- feedback
    def engagement(self, agent_id: str, keys: Sequence[str]) -> Dict[str, tuple]:
        if not keys:
            return {}
        rows = self.all(
            f"SELECT key, alpha, beta FROM engagement_stats WHERE agent_id = ? AND key IN ({self._placeholders(len(keys))})",
            [agent_id, *keys],
        )
        return {r["key"]: (r["alpha"], r["beta"]) for r in rows}

    def bump_engagement(self, agent_id: str, keys: Iterable[str], d_alpha: float, d_beta: float) -> None:
        with self.transaction():
            for key in keys:
                self._conn.execute(
                    """INSERT INTO engagement_stats (agent_id, key, alpha, beta) VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, key) DO UPDATE SET alpha = alpha + ?, beta = beta + ?""",
                    (agent_id, key, 1.0 + d_alpha, 1.0 + d_beta, d_alpha, d_beta),
                )

    # ----------------------------------------------------------------- reports
    def add_report(self, post_id: str, reporter_id: str, reason: str) -> bool:
        try:
            self.execute("INSERT INTO reports VALUES (?,?,?,?,?)", (new_id("rpt"), post_id, reporter_id, reason, now()))
            return True
        except sqlite3.IntegrityError:
            return False

    def count_reports(self, post_id: str) -> int:
        return self.scalar("SELECT COUNT(*) FROM reports WHERE post_id = ?", (post_id,))
