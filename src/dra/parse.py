"""Divide um ficheiro markdown do espelho LawAgents/pt em artigos.

Formato de entrada (verificado no espelho, commit fixado em config/corpus.yaml):

    ---
    id: "lei-7-2009-livro-i-..."
    dre_id: "602073"
    title: "Código do Trabalho (Lei n.º 7/2009, de 12 de fevereiro) — Livro I, ..."
    source: "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/2009-34546475"
    part_of: "lei-7-2009"            # só nas partes de diplomas grandes
    ...
    ---
    ## Anexo — CÓDIGO DO TRABALHO
    ### Livro I — Parte geral
    ###### Artigo 1.º — Fontes específicas     (ou "Artigo 58.º (Direito ao trabalho)")
    texto...
    - *Alterado pelo/a Artigo 2.º do/a [Lei n.º ...](https://...), em vigor a partir de ...*

O nível do título de um artigo varia (## a ######) conforme a profundidade da hierarquia; qualquer título que
não seja de artigo atualiza a hierarquia nesse nível.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

_ARTICLE_RE = re.compile(
    r"^(?P<hashes>#{2,6})\s+Artigo\s+(?P<num>\d+)\.?\s*º?(?P<suffix>(?:-[A-Z]+)*)"
    r"(?:\s*(?:—|-)\s*(?P<ep1>.+?)|\s*\((?P<ep2>.+)\))?\s*$"
)
_ARTICLE_UNICO_RE = re.compile(r"^(?P<hashes>#{2,6})\s+Artigo\s+único(?:\s*(?:—|-)\s*(?P<ep>.+))?\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<text>.+?)\s*$")
_NOTE_RE = re.compile(r"^-\s+\*(?P<note>(?:Alterad|Aditad|Revogad|Retificad|Rectificad|Republicad)[^*]*)\*\s*$")
# Artigo revogado no espelho: "*(Revogado)* — *Revogado pelo/a Artigo 54.º do/a Decreto-Lei n.º 84/2021 …*"
_REVOKED_RE = re.compile(r"^\*\((?P<w>Revogad[oa])\)\*\s*(?:—\s*\*(?P<note>.+)\*)?\s*$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


@dataclass
class Article:
    diploma_id: str          # id do diploma completo (part_of, ou o próprio id)
    dre_id: str
    diploma_title: str       # título oficial sem o sufixo da parte
    number: str              # "238.º", "1022.º-A", "único"
    number_int: int | None   # 238 (para ordenação e filtros)
    epigrafe: str
    hierarchy: list[str]     # ["Anexo — CÓDIGO DO TRABALHO", "Livro I — Parte geral", ...]
    scope: str               # "diploma" (diploma de aprovação) ou "anexo"/"anexo-iv" (código anexo)
    paragraphs: list[str]
    notes: list[str] = field(default_factory=list)
    url: str = ""
    last_amended: str = ""
    text_version: str = ""

    @property
    def in_annex(self) -> bool:
        return self.scope != "diploma"

    @property
    def key(self) -> str:
        return f"{self.diploma_id}:{self.scope}:art-{self.number.replace('.º', '')}"

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def label(self) -> str:
        """Ex.: 'Artigo 238.º — Duração do período de férias'."""
        return f"Artigo {self.number}" + (f" — {self.epigrafe}" if self.epigrafe else "")


def split_front_matter(raw: str) -> tuple[dict, str]:
    if not raw.startswith("---"):
        return {}, raw
    end = raw.find("\n---", 3)
    if end == -1:
        return {}, raw
    meta = yaml.safe_load(raw[3:end]) or {}
    return meta, raw[end + 4 :].lstrip("\n")


def _strip_part_suffix(title: str, part: str | None) -> str:
    if part and title.endswith(f" — {part}"):
        return title[: -len(f" — {part}")]
    return title


def annex_scope(path: list[str]) -> str:
    """'anexo' / 'anexo-iv' para artigos de um anexo; 'diploma' para os do próprio diploma."""
    for h in path:
        if h.lower().startswith("anexo"):
            head = h.split(" — ")[0].strip().lower()
            return re.sub(r"[^a-z0-9]+", "-", head).strip("-")
    return "diploma"


def parse_markdown(raw: str, hierarchy: dict[int, str] | None = None) -> list[Article]:
    """Extrai os artigos de um ficheiro.

    `hierarchy` é a pilha de títulos {nível: texto}. Os diplomas grandes estão divididos em partes consecutivas
    e só a primeira parte de um anexo traz o título "## Anexo …"; passando o mesmo dicionário a todas as partes,
    pela ordem do ficheiro de visão geral, cada artigo fica com a hierarquia completa e o anexo certo.
    """
    meta, body = split_front_matter(raw)
    diploma_id = str(meta.get("part_of") or meta.get("id") or "")
    title = _strip_part_suffix(str(meta.get("title", "")), meta.get("part"))
    base = {
        "diploma_id": diploma_id,
        "dre_id": str(meta.get("dre_id", "")),
        "diploma_title": title,
        "url": str(meta.get("source", "")),
        "last_amended": str(meta.get("last_amended", "") or ""),
        "text_version": str(meta.get("text_version", "") or ""),
    }
    if hierarchy is None:
        hierarchy = {}
    if str(meta.get("part") or "").lower().startswith("diploma de aprova"):
        hierarchy.clear()

    articles: list[Article] = []
    current: Article | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current, buf
        if current is not None:
            paragraphs, notes = _paragraphs(buf)
            current.paragraphs, current.notes = paragraphs, notes
            if paragraphs:
                articles.append(current)
        current, buf = None, []

    for line in body.splitlines():
        m = _ARTICLE_RE.match(line) or _ARTICLE_UNICO_RE.match(line)
        if m:
            flush()
            level = len(m.group("hashes"))
            path = [hierarchy[k] for k in sorted(hierarchy) if k < level]
            if "num" in m.groupdict() and m.group("num"):
                number = f"{m.group('num')}.º{m.group('suffix') or ''}"
                number_int = int(m.group("num"))
                ep = m.group("ep1") or m.group("ep2") or ""
            else:
                number, number_int, ep = "único", None, m.group("ep") or ""
            current = Article(
                number=number,
                number_int=number_int,
                epigrafe=ep.strip(),
                hierarchy=path,
                scope=annex_scope(path),
                paragraphs=[],
                **base,
            )
            continue
        h = _HEADING_RE.match(line)
        if h:
            flush()
            level = len(h.group("hashes"))
            if level == 1:
                continue  # título do ficheiro, não faz parte da hierarquia
            for k in [k for k in hierarchy if k >= level]:
                del hierarchy[k]
            hierarchy[level] = h.group("text")
            continue
        if current is not None:
            buf.append(line)
    flush()
    return articles


def _paragraphs(lines: list[str]) -> tuple[list[str], list[str]]:
    paragraphs: list[str] = []
    notes: list[str] = []
    cur: list[str] = []
    for line in lines + [""]:
        r = _REVOKED_RE.match(line.strip())
        if r:
            paragraphs.append(f"({r.group('w')})")
            if r.group("note"):
                notes.append(_LINK_RE.sub(r"\1", r.group("note")).strip())
            continue
        n = _NOTE_RE.match(line.strip())
        if n:
            notes.append(_LINK_RE.sub(r"\1", n.group("note")).strip())
            continue
        if line.strip():
            cur.append(line.strip())
        elif cur:
            paragraphs.append(" ".join(cur))
            cur = []
    return paragraphs, notes


def dedupe(articles: list[Article]) -> list[Article]:
    """Mesmo número no mesmo âmbito (raro: listas anexas com numeração própria) → sufixo [2], [3]…"""
    seen: dict[str, int] = {}
    for a in articles:
        k = a.key
        if k in seen:
            seen[k] += 1
            a.number = f"{a.number} [{seen[k]}]"
        else:
            seen[k] = 1
    return articles
