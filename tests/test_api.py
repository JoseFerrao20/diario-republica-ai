from fastapi.testclient import TestClient

from dra import api
from dra import app as app_mod
from dra.rag import Answerer
from tests.conftest import fake_client, response, text


def test_ask_endpoint_returns_citations_cost_and_latency(searcher, monkeypatch):
    client, _ = fake_client(lambda kw: response([text("22 dias úteis", [(0, 0, 1, "1 - O período…")])]))
    monkeypatch.setattr(api, "answerer", lambda db: Answerer(searcher, client=client, k=3))
    monkeypatch.setattr(api, "searcher", lambda db: searcher)
    r = TestClient(api.app).post("/api/ask", json={"question": "artigo 238.º do Código do Trabalho"})
    assert r.status_code == 200
    d = r.json()
    assert d["citations"][0]["url"].startswith("https://diariodarepublica.pt/")
    assert d["usage"]["input_tokens"] == 1000 and d["cost_usd"] > 0
    assert d["retrieval_ms"] > 0 and d["stats"]["n"] == 1
    assert TestClient(api.app).get("/api/stats").json()["n"] == 1


def test_index_page_is_served():
    r = TestClient(api.app).get("/")
    assert r.status_code == 200 and "Diário da República explicado" in r.text
    assert app_mod  # módulo importa sem índice construído
