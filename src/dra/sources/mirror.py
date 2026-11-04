"""Fonte principal: o espelho LawAgents/pt, fixado num commit.

Descarrega só as pastas/ficheiros do corpus (sparse checkout, sem histórico) e devolve os artigos.
As ligações citadas são sempre as oficiais do DRE (campo `source` de cada ficheiro), nunca as do espelho.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from dra.config import Corpus, DiplomaSpec
from dra.parse import Article, dedupe, parse_markdown


def _git(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def fetch(corpus: Corpus, dest: Path, repo_url: str | None = None) -> Path:
    """Garante uma cópia do espelho em `dest`, no commit fixado, com os caminhos do corpus."""
    repo_url = repo_url or corpus.mirror_repo
    patterns = [("/" + d.path.rstrip("/") + ("/" if d.path.endswith("/") else "")) for d in corpus.diplomas]
    if not (dest / ".git").exists():
        dest.mkdir(parents=True, exist_ok=True)
        _git("init", "-q", cwd=dest)
        _git("remote", "add", "origin", repo_url, cwd=dest)
    _git("sparse-checkout", "set", "--no-cone", *patterns, cwd=dest)
    _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", corpus.mirror_commit, cwd=dest)
    _git("checkout", "-q", "--force", corpus.mirror_commit, cwd=dest)
    return dest


_PART_ROW_RE = re.compile(r"^\|[^|]*\|\s*(?P<path>[^|]+\.md)\s*\|")


def files_for(spec: DiplomaSpec, root: Path) -> list[Path]:
    """Ficheiros de um diploma pela ordem do documento.

    Diplomas divididos têm um ficheiro de visão geral (`<id>.md`) com a tabela "parte | caminho | …" na ordem
    do texto; a ordem alfabética dos nomes não serve (ex.: "livro-i-titulo-ii" antes de "livro-i-titulo-i-a…").
    """
    p = root / spec.path
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise FileNotFoundError(f"{spec.path} não existe no espelho ({root})")
    overview = next((f for f in p.rglob(f"{spec.id}.md")), None)
    if overview is not None:
        ordered = [
            root / m.group("path").strip()
            for line in overview.read_text(encoding="utf-8").splitlines()
            if (m := _PART_ROW_RE.match(line))
        ]
        ordered = [f for f in ordered if f.is_file()]
        if ordered:
            return ordered
        return [overview]  # diploma numa pasta mas sem partes
    return sorted(p.rglob("*.md"))


def iter_articles(corpus: Corpus, root: Path) -> Iterator[tuple[DiplomaSpec, Article]]:
    for spec in corpus.diplomas:
        hierarchy: dict[int, str] = {}
        arts: list[Article] = []
        for f in files_for(spec, root):
            arts += [a for a in parse_markdown(f.read_text(encoding="utf-8"), hierarchy) if a.diploma_id == spec.id]
        for a in dedupe(arts):
            yield spec, a
