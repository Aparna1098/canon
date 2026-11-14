"""Canon's shared memory of verified facts (SQLite).

Only verified answers are stored. Each row holds the public fact and its source, plus anonymous counts.
No user identity is stored.

Lookup: embed the question (the classifier's MiniLM), take the nearest unexpired fact with cosine > 0.85
whose entity appears in the new question and whose years, quarters and version numbers match exactly.
That last check is what stops "Tesla revenue Q3 2026" from matching "Tesla revenue Q2 2026".
"""
from __future__ import annotations

import datetime as dt
import os
import re
import sqlite3
import threading

import numpy as np

from . import classifier

DB_PATH = os.environ.get("CANON_DB", os.path.join(os.path.dirname(os.path.dirname(__file__)), "facts.db"))
MIN_COSINE = 0.85
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    embedding BLOB NOT NULL,
    category TEXT NOT NULL,
    entity TEXT,
    answer TEXT NOT NULL,
    quote TEXT,
    source_url TEXT,
    source_type TEXT,
    verified_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    hit_count INTEGER NOT NULL DEFAULT 0,
    calls_saved INTEGER NOT NULL DEFAULT 0,
    calls_used INTEGER NOT NULL DEFAULT 0,
    tokens_read INTEGER NOT NULL DEFAULT 0,
    tokens_saved INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    monitor_id TEXT
)"""


def _db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(SCHEMA)
    return con


def _now():
    return dt.datetime.now(dt.timezone.utc)


def expires(category, now=None):
    """code 1 day, finance until the next quarter end, medical 30 days, local and general 7 days."""
    now = now or _now()
    if category == "finance":
        q_end_month = ((now.month - 1) // 3 + 1) * 3
        first_next = dt.datetime(now.year + (q_end_month == 12), q_end_month % 12 + 1, 1, tzinfo=dt.timezone.utc)
        return first_next - dt.timedelta(seconds=1)
    return now + dt.timedelta(days={"code": 1, "medical": 30}.get(category, 7))


def embed(text):
    model, _ = classifier._load()
    return model.encode([text], normalize_embeddings=True)[0].astype(np.float32)


_KEY = re.compile(r"\b(?:q[1-4]|fy\s?\d{2,4}|(?:19|20)\d\d|\d+\.\d+(?:\.\d+)?)\b", re.I)


def key_tokens(q):
    """Years, quarters, fiscal years and version numbers: must match exactly for a memory hit."""
    return {t.lower().replace(" ", "") for t in _KEY.findall(q)}


def entity_in(entity, question):
    if not entity:
        return True
    ql = question.lower()
    from .sources import KNOWN_PACKAGES  # code facts store the registry name ("torch"); questions say "pytorch"
    aliases = {a for a, (pkg, _) in KNOWN_PACKAGES.items() if pkg == entity.lower()}
    tokens = [t for t in re.findall(r"[a-z0-9.]+", entity.lower()) if len(t) > 2] or [entity.lower()]
    return any(t in ql for t in tokens) or any(re.search(rf"(?<![\w]){re.escape(a)}(?![\w])", ql) for a in aliases)


def lookup(question, emb=None):
    """Return (row, cosine) for the best valid memory hit, or (None, best_cosine)."""
    emb = embed(question) if emb is None else emb
    keys = key_tokens(question)
    now = _now().isoformat()
    with _lock, _db() as con:
        rows = con.execute("SELECT * FROM facts WHERE expires_at > ?", (now,)).fetchall()
    best, best_cos = None, 0.0
    for r in rows:
        cos = float(np.frombuffer(r["embedding"], dtype=np.float32) @ emb)
        if cos <= MIN_COSINE or cos <= best_cos:
            continue
        if not entity_in(r["entity"], question) or key_tokens(r["question"]) != keys:
            continue
        best, best_cos = r, cos
    return best, best_cos


def record_hit(fact_id, calls_saved, tokens_saved):
    with _lock, _db() as con:
        con.execute("UPDATE facts SET hit_count = hit_count + 1, calls_saved = calls_saved + ?, "
                    "tokens_saved = tokens_saved + ? WHERE id = ?", (calls_saved, tokens_saved, fact_id))


def store(question, emb, r, result_json):
    """Store a verified answer. Unverified answers are never stored."""
    if not r.get("verified"):
        return None
    now = _now()
    with _lock, _db() as con:
        cur = con.execute(
            "INSERT INTO facts (question, embedding, category, entity, answer, quote, source_url, source_type, "
            "verified_at, expires_at, calls_used, tokens_read, result_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (question, emb.astype(np.float32).tobytes(), r["category"], r.get("entity"), r["answer"], r.get("quote"),
             r.get("url"), r.get("source_type"), now.isoformat(), expires(r["category"], now).isoformat(),
             r["calls_used"], r["tokens_read"], result_json))
        return cur.lastrowid


def all_facts():
    with _lock, _db() as con:
        return [dict(r) for r in con.execute(
            "SELECT id, question, category, entity, answer, quote, source_url, source_type, verified_at, "
            "expires_at, hit_count, calls_saved, tokens_saved FROM facts ORDER BY hit_count DESC, id DESC")]


def stats():
    with _lock, _db() as con:
        r = con.execute("SELECT COUNT(*) n, COALESCE(SUM(hit_count),0) hits, COALESCE(SUM(calls_saved),0) calls, "
                        "COALESCE(SUM(tokens_saved),0) tokens FROM facts").fetchone()
    return {"facts_stored": r["n"], "memory_hits": r["hits"], "searches_saved": r["calls"], "tokens_saved": r["tokens"]}


def hot_sources(min_hits=3):
    with _lock, _db() as con:
        return [dict(r) for r in con.execute(
            "SELECT DISTINCT source_url, category FROM facts WHERE hit_count >= ? AND source_url IS NOT NULL",
            (min_hits,))]


def set_monitor(fact_id, monitor_id):
    with _lock, _db() as con:
        con.execute("UPDATE facts SET monitor_id = ? WHERE id = ?", (monitor_id, fact_id))
