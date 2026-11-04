"""Constrói o índice: artigos → SQLite (FTS5) → chunks → embeddings (sqlite-vec)."""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable

from dra.config import Corpus, DiplomaSpec
from dra.db import ensure_vec_table, set_meta
from dra.embed import Embedder
from dra.parse import Article

CHUNK_CHARS = 1200  # ~300 tokens: cabe no limite de 512 tokens dos modelos e5/MiniLM com o cabeçalho


def article_header(name: str, a: Article) -> str:
    section = a.hierarchy[-1] if a.hierarchy else ""
    return f"{name} — {a.label}" + (f" ({section})" if section else "")


def chunk_article(name: str, a: Article) -> list[str]:
    """Um chunk por artigo; artigos longos partidos entre números/parágrafos, sempre com o cabeçalho."""
    head = article_header(name, a)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for p in a.paragraphs:
        if cur and size + len(p) > CHUNK_CHARS:
            chunks.append(head + "\n" + "\n".join(cur))
            cur, size = [], 0
        # parágrafos gigantes (tabelas) são cortados; o texto completo continua no FTS e no contexto do LLM
        cur.append(p[: CHUNK_CHARS * 2])
        size += len(cur[-1])
    if cur:
        chunks.append(head + "\n" + "\n".join(cur))
    return chunks


def store_articles(conn: sqlite3.Connection, items: Iterable[tuple[DiplomaSpec, Article]]) -> int:
    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM articles_fts")
    conn.execute("DELETE FROM articles")
    conn.execute("DROP TABLE IF EXISTS vec_chunks")
    n = 0
    for spec, a in items:
        cur = conn.execute(
            """INSERT INTO articles (key, diploma_id, diploma_name, diploma_title, dre_id, number, number_int,
                   scope, epigrafe, hierarchy, paragraphs, notes, url, last_amended, text_version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                a.key, a.diploma_id, spec.name, a.diploma_title, a.dre_id, a.number, a.number_int, a.scope,
                a.epigrafe, " > ".join(a.hierarchy), json.dumps(a.paragraphs, ensure_ascii=False),
                json.dumps(a.notes, ensure_ascii=False), a.url, a.last_amended, a.text_version,
            ),
        )
        aid = cur.lastrowid
        conn.execute(
            "INSERT INTO articles_fts (rowid, diploma_name, epigrafe, hierarchy, body) VALUES (?,?,?,?,?)",
            (aid, " ".join([spec.name, *spec.acronyms]), a.epigrafe, " ".join(a.hierarchy), a.text),
        )
        for i, text in enumerate(chunk_article(spec.name, a)):
            conn.execute("INSERT INTO chunks (article_id, ord, text) VALUES (?,?,?)", (aid, i, text))
        n += 1
    conn.commit()
    return n


def embed_chunks(conn: sqlite3.Connection, embedder: Embedder, batch: int = 256, log=print) -> int:
    ensure_vec_table(conn, embedder.dim)
    rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        vecs = embedder.embed_passages([r["text"] for r in part])
        conn.executemany(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            [(r["id"], v.astype("float32").tobytes()) for r, v in zip(part, vecs, strict=True)],
        )
        if (i // batch) % 10 == 0:
            log(f"  embeddings: {i + len(part)}/{len(rows)} ({time.perf_counter() - t0:.0f}s)")
    set_meta(conn, embedder=embedder.name, dim=embedder.dim)
    conn.commit()
    return len(rows)


def build(conn: sqlite3.Connection, corpus: Corpus, items, embedder: Embedder | None, log=print) -> dict:
    t0 = time.perf_counter()
    n = store_articles(conn, items)
    set_meta(conn, mirror_commit=corpus.mirror_commit, embedder=None)
    log(f"{n} artigos guardados ({time.perf_counter() - t0:.1f}s)")
    nchunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    if embedder is not None:
        embed_chunks(conn, embedder, log=log)
    return {"articles": n, "chunks": nchunks, "seconds": round(time.perf_counter() - t0, 1)}
