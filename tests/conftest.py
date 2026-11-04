from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from dra import db, index
from dra.config import load_corpus
from dra.embed import HashingEmbedder
from dra.search import Searcher
from dra.sources import mirror

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(FIX / "corpus.yaml")


@pytest.fixture(scope="session")
def articles(corpus):
    return list(mirror.iter_articles(corpus, FIX / "mirror"))


@pytest.fixture()
def searcher(corpus, articles, tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    index.build(conn, corpus, articles, HashingEmbedder(256), log=lambda *_: None)
    return Searcher(conn, corpus, HashingEmbedder(256))


class FakeMessages:
    """Imita client.beta.messages: devolve uma resposta pré-definida e guarda o pedido."""

    def __init__(self, make_response):
        self.make_response = make_response
        self.calls: list[dict] = []

    def create(self, **kw):
        self.calls.append(kw)
        return self.make_response(kw)


def fake_client(make_response):
    msgs = FakeMessages(make_response)
    return NS(beta=NS(messages=msgs)), msgs


def response(blocks, stop_reason="end_turn", model="claude-opus-5-5", **usage):
    u = {"input_tokens": 1000, "output_tokens": 200, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    u.update(usage)
    return NS(content=blocks, stop_reason=stop_reason, model=model, usage=NS(**u))


def text(t, cites=()):
    return NS(type="text", text=t, citations=[
        NS(type="content_block_location", document_index=d, start_block_index=s, end_block_index=e, cited_text=c)
        for d, s, e, c in cites
    ] or None)
