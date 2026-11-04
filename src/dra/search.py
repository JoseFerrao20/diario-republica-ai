"""Pesquisa híbrida: BM25 (FTS5) + kNN denso (sqlite-vec), fundidos por Reciprocal Rank Fusion, mais referências
explícitas a artigos ("artigo 238.º do Código do Trabalho", "art. 1022.º CC") que vão diretamente para o topo."""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field

from dra.config import Corpus
from dra.embed import Embedder

RRF_K = 60

STOPWORDS = {"a", "ao", "aos", "as", "à", "às", "com", "como", "contra", "da", "das", "de", "dela", "dele", "deles", "do", "dos", "e", "é", "ela", "ele", "eles", "em", "entre", "era", "essa", "esse", "esta", "este", "eu", "foi", "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu", "minha", "muito", "na", "nas", "nem", "no", "nos", "não", "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual", "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "sua", "suas", "são", "só", "também", "te", "tem", "ter", "um", "uma", "umas", "uns", "vai", "vou", "posso", "pode", "podem", "devo", "deve", "quais", "quanto", "quantos", "quantas", "onde", "sobre", "meus", "minhas", "tenho", "estou", "está", "estão", "fazer", "feito", "caso", "lei", "artigo", "artigos", "código", "n", "nº", "º"}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def _stem(tok: str) -> str:
    """Radical grosseiro para prefix queries do FTS5 (o unicode61 não tem stemmer para português)."""
    for suf in ("mente", "ções", "coes", "ção", "cao", "idades", "idade", "mentos", "mento", "ados", "adas",
                "ando", "endo", "ar", "er", "ir", "as", "os", "es", "a", "o", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def fts_query(question: str) -> str | None:
    toks = [t for t in re.findall(r"\w+", strip_accents(question.lower())) if t not in STOPWORDS and len(t) > 1]
    toks = [t for t in toks if not t.isdigit()]
    if not toks:
        return None
    seen: list[str] = []
    for t in toks:
        term = f'"{_stem(t)}"*' if len(t) > 4 else f'"{t}"'
        if term not in seen:
            seen.append(term)
    return " OR ".join(seen)


@dataclass
class Hit:
    article_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # "bm25" / "dense" / "ref" → posição (1 = melhor)
    similarity: float | None = None                       # cosseno do melhor chunk, se houver pesquisa densa


@dataclass
class SearchResult:
    hits: list[Hit]
    explicit_refs: list[int]
    best_similarity: float | None


class Searcher:
    def __init__(self, conn: sqlite3.Connection, corpus: Corpus, embedder: Embedder | None = None):
        self.conn = conn
        self.corpus = corpus
        self.embedder = embedder
        self._aliases = self._build_aliases()
        self._main_scope = {
            r["diploma_id"]: r["scope"]
            for r in conn.execute(
                """SELECT diploma_id, scope FROM (
                       SELECT diploma_id, scope, count(*) AS n,
                              row_number() OVER (PARTITION BY diploma_id ORDER BY count(*) DESC) AS rn
                       FROM articles GROUP BY diploma_id, scope) WHERE rn = 1"""
            )
        }

    # ---- referências explícitas -------------------------------------------------------------------------

    def _build_aliases(self) -> list[tuple[re.Pattern, str]]:
        out = []
        for d in self.corpus.diplomas:
            for alias in [d.name, *d.acronyms]:
                a = re.escape(strip_accents(alias.lower())).replace(r"\ ", r"\s+")
                out.append((re.compile(rf"(?<!\w){a}(?!\w)"), d.id))
        # aliases longos primeiro: "Código do IVA" antes de "IVA"
        return sorted(out, key=lambda x: -len(x[0].pattern))

    _ART_RE = re.compile(r"\bart(?:igo|\.?\s*º|\.)?s?\s*(\d+)\s*\.?\s*º?\s*(-\s*[a-z]+)?", re.IGNORECASE)

    def explicit_refs(self, question: str) -> list[int]:
        q = strip_accents(question.lower())
        found: list[int] = []
        for m in self._ART_RE.finditer(q):
            num = int(m.group(1))
            suffix = (m.group(2) or "").replace(" ", "").upper()
            window = q[m.end() : m.end() + 80]
            diploma = next((did for pat, did in self._aliases if pat.search(window)), None)
            if diploma is None:
                continue  # sem diploma não arriscamos: "artigo 13.º" existe em todos
            rows = self.conn.execute(
                "SELECT id, scope, number FROM articles WHERE diploma_id = ? AND number_int = ?", (diploma, num)
            ).fetchall()
            rows = [r for r in rows if r["number"].replace(".º", "").endswith(suffix) or not suffix]
            rows = [r for r in rows if (r["number"].count("-") > 0) == bool(suffix)] or rows
            main = self._main_scope.get(diploma)
            rows.sort(key=lambda r: (r["scope"] != main, r["scope"] == "diploma"))
            if rows and rows[0]["id"] not in found:
                found.append(rows[0]["id"])
        return found

    # ---- pesquisas base ---------------------------------------------------------------------------------

    def bm25(self, question: str, k: int = 50) -> list[int]:
        q = fts_query(question)
        if not q:
            return []
        rows = self.conn.execute(
            """SELECT rowid FROM articles_fts WHERE articles_fts MATCH ?
               ORDER BY bm25(articles_fts, 1.0, 4.0, 1.5, 1.0) LIMIT ?""",
            (q, k),
        ).fetchall()
        return [r[0] for r in rows]

    def dense(self, question: str, k: int = 50) -> list[tuple[int, float]]:
        if self.embedder is None:
            return []
        qv = self.embedder.embed_query(question).astype("float32").tobytes()
        rows = self.conn.execute(
            """SELECT c.article_id, v.distance FROM vec_chunks v JOIN chunks c ON c.id = v.rowid
               WHERE v.embedding MATCH ? AND k = ? ORDER BY v.distance""",
            (qv, k * 3),
        ).fetchall()
        best: dict[int, float] = {}
        for aid, dist in rows:
            if aid not in best:
                best[aid] = 1.0 - dist  # distância cosseno → semelhança
        return sorted(best.items(), key=lambda x: -x[1])[:k]

    # ---- híbrida ----------------------------------------------------------------------------------------

    def search(self, question: str, k: int = 8, mode: str = "hybrid") -> SearchResult:
        refs = self.explicit_refs(question)
        lists: dict[str, list[int]] = {}
        sims: dict[int, float] = {}
        if mode in ("hybrid", "bm25"):
            lists["bm25"] = self.bm25(question)
        if mode in ("hybrid", "dense"):
            d = self.dense(question)
            lists["dense"] = [aid for aid, _ in d]
            sims = dict(d)
        hits: dict[int, Hit] = {}
        for name, ids in lists.items():
            for rank, aid in enumerate(ids, start=1):
                h = hits.setdefault(aid, Hit(aid, 0.0, similarity=sims.get(aid)))
                h.score += 1.0 / (RRF_K + rank)
                h.ranks[name] = rank
        for rank, aid in enumerate(refs, start=1):
            h = hits.setdefault(aid, Hit(aid, 0.0, similarity=sims.get(aid)))
            h.score += 1.0 / rank  # domina qualquer pontuação RRF (máx. 2/61)
            h.ranks["ref"] = rank
        ordered = sorted(hits.values(), key=lambda h: -h.score)[:k]
        best_sim = max(sims.values()) if sims else None
        return SearchResult(ordered, refs, best_sim)

    # ---- leitura ----------------------------------------------------------------------------------------

    def article(self, article_id: int) -> dict:
        r = self.conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        d = dict(r)
        d["paragraphs"] = json.loads(d["paragraphs"])
        d["notes"] = json.loads(d["notes"] or "[]")
        return d
