"""Adaptador direto ao DRE.

`fixtures/dre/lei-24-1996.json.gz` é a resposta real de `DataActionGetData` para a Lei 24/96 (DiplomaLegisId 406882),
gravada pelo projeto LawAgents/pt (scripts/tests/fixtures/raw, licença MIT). `lei-24-1996.espelho.md` é a mesma lei
no espelho LawAgents/pt, no commit fixado em config/corpus.yaml. Os dois têm de dar os mesmos artigos.
"""

import gzip
import itertools
import json
import re
from pathlib import Path

import pytest

from dra import db, index
from dra.embed import HashingEmbedder
from dra.parse import parse_markdown
from dra.sources.dre import USER_AGENT, DREClient, FetchAborted, ServiceChanged, articles_from_consolidated

FIX = Path(__file__).parent / "fixtures" / "dre"
with gzip.open(FIX / "lei-24-1996.json.gz") as _f:
    RAW = json.load(_f)
URL = "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/1996-34491075"


def convert(data=None):
    return articles_from_consolidated(data or RAW["data"], diploma_id="lei-24-1996", dre_id="406882", diploma_title="Lei n.º 24/96",
                                      url=URL)


def norm(ps):
    return [" ".join(p.split()) for p in ps]


# ---- conversão ----------------------------------------------------------------------------------------------


def test_real_response_matches_the_mirror_article_by_article():
    dre = {a.key: a for a in convert()}
    mirror = {a.key: a for a in parse_markdown((FIX / "lei-24-1996.espelho.md").read_text(encoding="utf-8"))}
    assert dre.keys() == mirror.keys()
    assert len(dre) == 29
    for k, a in dre.items():
        assert norm(a.paragraphs) == norm(mirror[k].paragraphs), k
        assert a.epigrafe == mirror[k].epigrafe, k
        assert a.hierarchy == mirror[k].hierarchy, k


def test_revoked_articles_never_carry_their_old_text():
    # O DRE mantém o texto anterior em `Texto` nos fragmentos revogados (VersaoEstadoId 4).
    raw_9b = next(x for x in RAW["data"]["LegConsBase"]["List"] if x["FragmentoVersao"]["Tituo"] == "Artigo 9.º-B")
    assert raw_9b["FragmentoVersao"]["Texto"].startswith("1 - O fornecedor de bens deve entregar")
    a = {a.key: a for a in convert()}["lei-24-1996:diploma:art-9-B"]
    assert a.paragraphs == ["(Revogado)"]
    assert a.notes[0].startswith("Revogado pelo/a Artigo 54.º do/a Decreto-Lei n.º 84/2021")
    assert "[" not in a.notes[0]


def test_metadata():
    a = convert()[0]
    assert (a.dre_id, a.number, a.epigrafe, a.url) == ("406882", "1.º", "Dever geral de proteção", URL)
    assert a.hierarchy == ["Capítulo I — Princípios gerais"]


def test_unknown_shape_raises_instead_of_guessing():
    with pytest.raises(ServiceChanged):
        convert({"Outra": 1})
    with pytest.raises(ServiceChanged):
        convert({"LegConsBase": {"List": [{"Nivel": "1"}]}})


# ---- cliente (transporte simulado com as formas reais dos serviços) ----------------------------------------


class FakeDRE:
    """Responde como o DRE: moduleinfo → JS do ecrã com o apiVersion → screen service com `data`."""

    def __init__(self, data=None):
        self.data = data or RAW["data"]
        self.calls: list[tuple[str, dict, dict | None]] = []
        self.fault = None  # callable(n, url) -> None | int | "oserror" | "garbage" | dict
        self.version_changes = 0

    def __call__(self, url, body, headers):
        self.calls.append((url, headers, json.loads(body) if body else None))
        if self.fault and (f := self.fault(len(self.calls), url)) is not None:
            if f == "oserror":
                raise OSError("rede em baixo")
            if f == "garbage":
                return 200, b"<html>mudou</html>"
            if isinstance(f, dict):
                return 200, json.dumps(f).encode()
            return f, b""
        if url.endswith("moduleservices/moduleinfo"):
            return 200, json.dumps({"manifest": {"versionToken": "tok-1", "urlVersions": {
                "/dr/scripts/dr.LegislacaoConsolidada.LegCons_Detalhe.mvc.js": "?v=42"}}}).encode()
        if re.search(r"/dr/scripts/dr\.\w+\.\w+\.mvc\.js", url):
            return 200, b'x;controller.callDataAction("DataActionGetData", "screenservices/x", "id", "api-777", f)'
        if self.version_changes:
            self.version_changes -= 1
            return 200, json.dumps({"versionInfo": {"hasModuleVersionChanged": True}}).encode()
        return 200, json.dumps({"versionInfo": {}, "data": self.data}).encode()


class Clock:
    def __init__(self):
        self.t = 0.0
        self.starts: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def client(fake, clock=None):
    clock = clock or Clock()
    return DREClient(transport=fake, clock=clock, sleep=clock.sleep, log=lambda *_: None)


def test_consolidated_request_protocol():
    fake = FakeDRE()
    data = client(fake).consolidated("406882")
    assert data == RAW["data"]
    urls = [u for u, _, _ in fake.calls]
    assert urls[0].endswith("/dr/moduleservices/moduleinfo")
    assert urls[1].endswith("/dr/scripts/dr.LegislacaoConsolidada.LegCons_Detalhe.mvc.js?v=42")
    assert urls[2].endswith("/dr/screenservices/dr/LegislacaoConsolidada/LegCons_Detalhe/DataActionGetData")
    _url, headers, body = fake.calls[2]
    assert body["versionInfo"] == {"moduleVersion": "tok-1", "apiVersion": "api-777"}
    assert body["screenData"]["variables"]["DiplomaLegisId"] == "406882"
    assert headers["X-CSRFToken"] and headers["User-Agent"] == USER_AGENT
    assert all(h["User-Agent"] == USER_AGENT for _, h, _ in fake.calls)


def test_requests_are_spaced_out():
    fake, clock = FakeDRE(), Clock()
    starts = []
    orig = fake.__call__

    def timed(url, body, headers):
        starts.append(clock.t)
        return orig(url, body, headers)

    c = DREClient(transport=timed, clock=clock, sleep=clock.sleep, log=lambda *_: None)
    c.consolidated(1)
    c.consolidated(2)
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert gaps and min(gaps) >= 0.55 - 1e-9


def test_redeploy_refreshes_version_tokens():
    fake = FakeDRE()
    fake.version_changes = 1
    assert client(fake).consolidated(1) == RAW["data"]
    assert sum(u.endswith("moduleinfo") for u, _, _ in fake.calls) == 2


@pytest.mark.parametrize("status", [429, 503])
def test_stops_immediately_when_asked_to_slow_down(status):
    fake = FakeDRE()
    fake.fault = lambda n, url: status if "screenservices" in url else None
    with pytest.raises(FetchAborted):
        client(fake).consolidated(1)
    assert sum("screenservices" in u for u, _, _ in fake.calls) == 1


def test_stops_after_three_consecutive_errors():
    fake = FakeDRE()
    fake.fault = lambda n, url: "oserror" if "screenservices" in url else None
    with pytest.raises(FetchAborted, match="3 erros seguidos"):
        client(fake).consolidated(1)


def test_transient_error_is_retried():
    fake = FakeDRE()
    fake.fault = lambda n, url: 500 if n == 3 else None
    assert client(fake).consolidated(1) == RAW["data"]


def test_site_change_is_reported():
    fake = FakeDRE()
    fake.fault = lambda n, url: "garbage" if url.endswith("moduleinfo") else None
    with pytest.raises(ServiceChanged):
        client(fake).consolidated(1)


def test_missing_api_version_is_reported():
    fake = FakeDRE()

    def no_api(url, body, headers):
        return (200, b"// outro javascript") if "mvc.js" in url else fake(url, body, headers)

    with pytest.raises(ServiceChanged, match="apiVersion"):
        client(no_api).consolidated(1)


# ---- substituição no índice ---------------------------------------------------------------------------------


def test_replace_diploma_updates_text_fts_and_vectors(corpus, articles, tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    emb = HashingEmbedder(256)
    index.build(conn, corpus, articles, emb, log=lambda *_: None)
    spec = corpus.by_id("lei-24-1996")
    ct_before = conn.execute("SELECT count(*) FROM articles WHERE diploma_id = 'lei-7-2009'").fetchone()[0]

    n = index.replace_diploma(conn, spec, convert(), emb)

    assert n == 29
    assert conn.execute("SELECT count(*) FROM articles WHERE diploma_id = 'lei-24-1996'").fetchone()[0] == 29
    assert conn.execute("SELECT count(*) FROM articles WHERE diploma_id = 'lei-7-2009'").fetchone()[0] == ct_before
    # FTS e vetores ficam coerentes: um vetor por chunk, nenhum órfão
    assert conn.execute("SELECT count(*) FROM articles_fts").fetchone()[0] == \
        conn.execute("SELECT count(*) FROM articles").fetchone()[0]
    assert conn.execute("SELECT count(*) FROM vec_chunks").fetchone()[0] == \
        conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    hit = conn.execute("SELECT a.key FROM articles_fts f JOIN articles a ON a.id = f.rowid "
                       "WHERE articles_fts MATCH 'ação inibitória'").fetchone()
    assert hit["key"] == "lei-24-1996:diploma:art-10"


def test_dre_sync_command_reports_and_applies_an_amendment(corpus, articles, tmp_path, monkeypatch, capsys):
    import copy

    from dra import cli, config
    from dra.sources import dre as dre_mod

    conn = db.connect(tmp_path / "t.sqlite")
    index.build(conn, corpus, articles, HashingEmbedder(256), log=lambda *_: None)
    conn.close()
    amended = copy.deepcopy(RAW["data"])
    art4 = next(x for x in amended["LegConsBase"]["List"] if x["FragmentoVersao"]["Tituo"] == "Artigo 4.º")
    art4["FragmentoVersao"]["Texto"] = "Nova redação do artigo 4.º."
    monkeypatch.setattr(dre_mod, "DREClient", lambda: client(FakeDRE(amended)))
    monkeypatch.setattr(cli, "load_corpus", lambda: corpus)
    monkeypatch.setattr(config, "DEFAULT_CORPUS", FIX.parent / "corpus.yaml")

    cli.main(["--db", str(tmp_path / "t.sqlite"), "dre-sync", "lei-24-1996"])
    out = capsys.readouterr().out
    # o índice de teste só tem 4 artigos da Lei 24/96 (excerto: 1.º, 2.º, 4.º, 12.º); o DRE tem 29
    assert "29 artigos no DRE, 4 no índice" in out
    assert "alterados: 1 — art-4" in out  # 1.º, 2.º e 12.º coincidem: sem falsos positivos

    cli.main(["--db", str(tmp_path / "t.sqlite"), "dre-sync", "lei-24-1996", "--apply"])
    conn = db.connect(tmp_path / "t.sqlite")
    row = conn.execute("SELECT paragraphs FROM articles WHERE key = 'lei-24-1996:diploma:art-4'").fetchone()
    assert json.loads(row["paragraphs"]) == ["Nova redação do artigo 4.º."]
    assert conn.execute("SELECT count(*) FROM articles WHERE diploma_id = 'lei-24-1996'").fetchone()[0] == 29
