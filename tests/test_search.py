from dra.search import fts_query


def key(searcher, hit):
    return searcher.article(hit.article_id)["key"]


def test_fts_query_drops_stopwords_and_accents():
    q = fts_query("Quantos dias de férias tenho por ano?")
    assert '"ferias"*' in q or '"feri"*' in q
    assert '"de"' not in q and "tenho" not in q


def test_explicit_reference_goes_first(searcher):
    r = searcher.search("O que diz o artigo 238.º do Código do Trabalho?", k=3)
    assert key(searcher, r.hits[0]) == "lei-7-2009:anexo:art-238"
    assert r.hits[0].ranks.get("ref") == 1


def test_explicit_reference_by_acronym_prefers_the_code_over_the_approving_law(searcher):
    r = searcher.search("art. 1.º do CT", k=3)
    assert key(searcher, r.hits[0]) == "lei-7-2009:anexo:art-1"


def test_reference_without_diploma_is_not_guessed(searcher):
    assert searcher.explicit_refs("o que diz o artigo 1.º?") == []


def test_bm25_finds_holiday_article(searcher):
    ids = searcher.bm25("período anual de férias dias úteis")
    keys = [searcher.article(i)["key"] for i in ids[:2]]
    assert "lei-7-2009:anexo:art-238" in keys


def test_hybrid_combines_both_lists(searcher):
    r = searcher.search("aviso prévio denúncia do contrato pelo trabalhador", k=5)
    assert any("bm25" in h.ranks and "dense" in h.ranks for h in r.hits)
    assert "lei-7-2009:anexo:art-400" in [key(searcher, h) for h in r.hits[:3]]


def test_no_results_for_empty_query(searcher):
    assert searcher.search("de a o", k=5, mode="bm25").hits == []
