import pytest

from dra.rag import REFUSAL_MARK, Answerer
from tests.conftest import fake_client, response, text


def test_request_sends_articles_as_citable_documents(searcher):
    client, msgs = fake_client(lambda kw: response([text("22 dias úteis.", [(0, 0, 1, "1 - O período…")])]))
    Answerer(searcher, client=client, k=3).answer("artigo 238.º do Código do Trabalho")
    kw = msgs.calls[0]
    assert kw["model"] == "claude-opus-5-5"
    assert kw["thinking"] == {"type": "adaptive"}
    assert "temperature" not in kw
    docs = [b for b in kw["messages"][0]["content"] if b["type"] == "document"]
    assert len(docs) == 3
    d0 = docs[0]
    assert d0["citations"] == {"enabled": True}
    assert d0["source"]["type"] == "content"
    assert d0["title"] == "Código do Trabalho — Artigo 238.º — Duração do período de férias"
    assert "diariodarepublica.pt" in d0["context"]
    assert kw["messages"][0]["content"][-1]["text"].startswith("Pergunta:")
    # prompt de sistema estável e em cache
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_citations_map_to_article_and_official_link(searcher):
    def make(kw):
        return response(
            [
                text("O período mínimo de férias é de "),
                text("22 dias úteis", [(0, 0, 1, "1 - O período anual de férias tem a duração mínima de 22 dias úteis.")]),
                text(".", []),
            ],
            input_tokens=3000, output_tokens=400, cache_read_input_tokens=1000,
        )

    client, _ = fake_client(make)
    ans = Answerer(searcher, client=client, k=3).answer("artigo 238.º do Código do Trabalho")
    assert not ans.refused
    assert len(ans.citations) == 1
    c = ans.citations[0]
    assert c.article_key == "lei-7-2009:anexo:art-238"
    assert c.url.startswith("https://diariodarepublica.pt/dr/legislacao-consolidada/")
    assert ans.segments[1].citations == [0]
    # custo: 3000×4 + 400×20 + 1000×0,20 por milhão
    assert ans.cost_usd == pytest.approx((3000 * 4 + 400 * 20 + 1000 * 0.2) / 1e6)
    assert ans.retrieval_ms > 0


def test_model_refusal_marker(searcher):
    client, _ = fake_client(lambda kw: response([text(f"{REFUSAL_MARK}: os artigos não falam de impostos.")]))
    ans = Answerer(searcher, client=client).answer("Qual a taxa de IRC?")
    assert ans.refused and "impostos" in ans.refusal_reason
    assert ans.citations == []


def test_answer_without_citations_is_treated_as_refusal(searcher):
    client, _ = fake_client(lambda kw: response([text("Acho que são 25 dias.")]))
    ans = Answerer(searcher, client=client).answer("Quantos dias de férias?")
    assert ans.refused


def test_safety_refusal_stop_reason(searcher):
    client, _ = fake_client(lambda kw: response([], stop_reason="refusal"))
    ans = Answerer(searcher, client=client).answer("férias")
    assert ans.refused and ans.usage is not None


def test_refuses_before_calling_llm_when_nothing_retrieved(searcher):
    # Só acontece sem pesquisa densa (o kNN devolve sempre vizinhos); com embeddings a barreira é o limiar.
    searcher.embedder = None
    client, msgs = fake_client(lambda kw: pytest.fail("não devia chamar o LLM"))
    ans = Answerer(searcher, client=client).answer("de a o")
    assert ans.refused and msgs.calls == [] and ans.cost_usd == 0


def test_refuses_before_llm_when_similarity_below_threshold(searcher):
    client, msgs = fake_client(lambda kw: pytest.fail("não devia chamar o LLM"))
    ans = Answerer(searcher, client=client, min_similarity=0.99).answer("receita de bacalhau com natas")
    assert ans.refused and msgs.calls == []
