from dra.parse import parse_markdown

SAMPLE = """---
id: "lei-1-2020"
dre_id: "123"
title: "Lei n.º 1/2020, de 1 de janeiro"
source: "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/2020-1"
---

# Lei n.º 1/2020, de 1 de janeiro

Preâmbulo que não é artigo.

## Capítulo I — Geral

### Artigo 1.º — Objeto

A presente lei regula coisas.

- *Alterado pelo/a Artigo 2.º do/a [Lei n.º 5/2021](https://diariodarepublica.pt/x), em vigor a partir de 2021-01-01*

### Artigo 2.º-A (Definições)

1 - Para efeitos da presente lei:

a) «Coisa» é uma coisa;

b) «Outra» é outra.

2 - Mais um número.

## Anexo — REGULAMENTO

### Artigo 1.º — Âmbito

Texto do anexo.
"""


def test_articles_epigraphs_and_paragraphs():
    arts = parse_markdown(SAMPLE)
    assert [(a.number, a.epigrafe, a.scope) for a in arts] == [
        ("1.º", "Objeto", "diploma"),
        ("2.º-A", "Definições", "diploma"),
        ("1.º", "Âmbito", "anexo"),
    ]
    assert arts[1].paragraphs == [
        "1 - Para efeitos da presente lei:", "a) «Coisa» é uma coisa;", "b) «Outra» é outra.", "2 - Mais um número.",
    ]
    assert arts[1].number_int == 2
    assert arts[0].hierarchy == ["Capítulo I — Geral"]


def test_amendment_notes_are_metadata_not_text():
    a = parse_markdown(SAMPLE)[0]
    assert a.paragraphs == ["A presente lei regula coisas."]
    assert a.notes == ["Alterado pelo/a Artigo 2.º do/a Lei n.º 5/2021, em vigor a partir de 2021-01-01"]


def test_keys_distinguish_diploma_from_annex():
    keys = [a.key for a in parse_markdown(SAMPLE)]
    assert keys == ["lei-1-2020:diploma:art-1", "lei-1-2020:diploma:art-2-A", "lei-1-2020:anexo:art-1"]


def test_metadata_and_official_link():
    a = parse_markdown(SAMPLE)[0]
    assert a.dre_id == "123"
    assert a.url.startswith("https://diariodarepublica.pt/")


def test_hierarchy_carries_across_parts_of_a_split_code(articles):
    by_key = {a.key: a for _, a in articles}
    # Parte 3 do Código do Trabalho não repete "## Anexo"; a hierarquia vem da parte anterior.
    a = by_key["lei-7-2009:anexo:art-238"]
    assert a.epigrafe == "Duração do período de férias"
    assert a.hierarchy[0].startswith("Anexo")
    assert a.paragraphs[0].startswith("1 - O período anual de férias tem a duração mínima de 22 dias úteis")
    # artigos 1.º do diploma de aprovação e 1.º do código anexo não colidem
    assert "lei-7-2009:diploma:art-1" in by_key and "lei-7-2009:anexo:art-1" in by_key
