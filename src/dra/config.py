from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "config" / "corpus.yaml"
DATA_DIR = Path(os.environ.get("DRA_DATA_DIR", ROOT / "data"))
DEFAULT_DB = Path(os.environ.get("DRA_DB", DATA_DIR / "dra.sqlite"))

# Modelos (ver docs/PLANO.md). O de resposta e o juiz podem ser trocados por variável de ambiente.
ANSWER_MODEL = os.environ.get("DRA_ANSWER_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.environ.get("DRA_JUDGE_MODEL", "claude-haiku-5-5")
ANSWER_EFFORT = os.environ.get("DRA_ANSWER_EFFORT", "medium")

# Embeddings: "fastembed:<modelo>", "voyage:<modelo>" ou "hashing" (só testes; não é semântico).
EMBEDDER = os.environ.get("DRA_EMBEDDER", "fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


@dataclass
class DiplomaSpec:
    id: str
    name: str
    path: str
    acronyms: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    mirror_repo: str
    mirror_commit: str
    diplomas: list[DiplomaSpec]

    def by_id(self, diploma_id: str) -> DiplomaSpec | None:
        return next((d for d in self.diplomas if d.id == diploma_id), None)


def load_corpus(path: Path = DEFAULT_CORPUS) -> Corpus:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Corpus(
        mirror_repo=raw["espelho"]["repositorio"],
        mirror_commit=raw["espelho"]["commit"],
        diplomas=[
            DiplomaSpec(id=d["id"], name=d["nome"], path=d["caminho"], acronyms=list(d.get("siglas", [])))
            for d in raw["diplomas"]
        ],
    )
