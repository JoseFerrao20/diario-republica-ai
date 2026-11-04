"""Esquema SQLite: artigos, índice BM25 (FTS5), chunks com vetores (sqlite-vec) e registo de respostas."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    key TEXT UNIQUE NOT NULL,
    diploma_id TEXT NOT NULL,
    diploma_name TEXT NOT NULL,
    diploma_title TEXT NOT NULL,
    dre_id TEXT,
    number TEXT NOT NULL,
    number_int INTEGER,
    scope TEXT NOT NULL,
    epigrafe TEXT,
    hierarchy TEXT,
    paragraphs TEXT NOT NULL,      -- JSON: lista de números/parágrafos (blocos citáveis)
    notes TEXT,                    -- JSON: notas de alteração
    url TEXT,
    last_amended TEXT,
    text_version TEXT
);
CREATE INDEX IF NOT EXISTS articles_lookup ON articles (diploma_id, number_int);

-- BM25. rowid = articles.id. Sem acentos (remove_diacritics 2): "ferias" encontra "férias".
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    diploma_name, epigrafe, hierarchy, body,
    tokenize = "unicode61 remove_diacritics 2"
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL REFERENCES articles(id),
    ord INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_article ON chunks (article_id);

CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    question TEXT NOT NULL,
    refused INTEGER NOT NULL,
    refusal_reason TEXT,
    model TEXT,
    input_tokens INTEGER, output_tokens INTEGER,
    cache_read_tokens INTEGER, cache_write_tokens INTEGER,
    cost_usd REAL,
    retrieval_ms REAL, generation_ms REAL, total_ms REAL,
    retrieved TEXT,                -- JSON: chaves dos artigos enviados ao modelo
    cited TEXT,                    -- JSON: chaves dos artigos citados
    answer TEXT
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.executescript(SCHEMA)
    return conn


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}] distance_metric=cosine)"
    )


def set_meta(conn: sqlite3.Connection, **kv: object) -> None:
    conn.executemany(
        "INSERT INTO meta(k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        [(k, json.dumps(v)) for k, v in kv.items()],
    )


def get_meta(conn: sqlite3.Connection, k: str, default: object = None) -> object:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (k,)).fetchone()
    return json.loads(row["v"]) if row else default


def has_vectors(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'vec_chunks'").fetchone() is not None
        and get_meta(conn, "embedder") is not None
    )


def log_run(conn: sqlite3.Connection, answer) -> int:
    """Guarda tokens, custo e latência de uma resposta (dra.rag.Answer)."""
    u = answer.usage
    cur = conn.execute(
        """INSERT INTO runs (question, refused, refusal_reason, model, input_tokens, output_tokens,
               cache_read_tokens, cache_write_tokens, cost_usd, retrieval_ms, generation_ms, total_ms,
               retrieved, cited, answer)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            answer.question, int(answer.refused), answer.refusal_reason, answer.model,
            u.input_tokens if u else 0, u.output_tokens if u else 0,
            u.cache_read_tokens if u else 0, u.cache_write_tokens if u else 0,
            answer.cost_usd, answer.retrieval_ms, answer.generation_ms, answer.total_ms,
            json.dumps([r["key"] for r in answer.retrieved]),
            json.dumps(sorted({c.article_key for c in answer.citations})),
            answer.text,
        ),
    )
    conn.commit()
    return cur.lastrowid


def run_stats(conn: sqlite3.Connection) -> dict:
    r = conn.execute(
        """SELECT count(*) AS n, avg(cost_usd) AS cost, avg(total_ms) AS ms,
                  avg(input_tokens) AS tin, avg(output_tokens) AS tout, sum(refused) AS refused
           FROM runs"""
    ).fetchone()
    return dict(r)
