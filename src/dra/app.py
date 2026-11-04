"""Componentes partilhados pela CLI, pela API e pela avaliação."""

from __future__ import annotations

from functools import lru_cache

from dra import db
from dra.config import DEFAULT_DB, load_corpus
from dra.embed import make_embedder
from dra.rag import Answerer
from dra.search import Searcher


@lru_cache(maxsize=4)
def searcher(db_path: str = str(DEFAULT_DB)) -> Searcher:
    conn = db.connect(db_path)
    emb = None
    name = db.get_meta(conn, "embedder")
    if name:  # a pergunta tem de usar o mesmo modelo que indexou os artigos
        emb = make_embedder(name)
    return Searcher(conn, load_corpus(), emb)


def answerer(db_path: str = str(DEFAULT_DB), **kw) -> Answerer:
    return Answerer(searcher(db_path), **kw)
