"""Embeddings. A Anthropic não tem API de embeddings; as opções são um modelo multilingue local (fastembed/ONNX,
custo zero, reprodutível no CI) ou a Voyage AI (parceira recomendada pela Anthropic).

`hashing` NÃO é um modelo semântico: são n-gramas de caracteres projetados num vetor. Existe só para os testes
e para correr o pipeline sem descarregar modelos. Nunca é usado para os resultados publicados.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return (m / np.where(n == 0, 1, n)).astype(np.float32)


# Prefixos que certos modelos exigem (o fastembed não os acrescenta: só os documenta).
_PREFIXES = {
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "google/embeddinggemma": ("task: search result | query: ", "title: none | text: "),
}


class FastEmbedEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # dependência opcional: pip install '.[local-embeddings]'

        self.name = f"fastembed:{model}"
        self._model = TextEmbedding(model_name=model)
        self._q, self._p = next((v for k, v in _PREFIXES.items() if model.startswith(k)), ("", ""))
        self.dim = len(self.embed_query("teste"))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed([self._p + t for t in texts], batch_size=64))))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(np.array(list(self._model.embed([self._q + text])))[0])


class VoyageEmbedder:
    def __init__(self, model: str):
        import voyageai  # dependência opcional: pip install '.[voyage]'; lê VOYAGE_API_KEY

        self.name = f"voyage:{model}"
        self._model = model
        self._client = voyageai.Client()
        self.dim = len(self.embed_query("teste"))

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 128):
            out += self._client.embed(texts[i : i + 128], model=self._model, input_type="document").embeddings
        return _normalize(np.array(out))

    def embed_query(self, text: str) -> np.ndarray:
        r = self._client.embed([text], model=self._model, input_type="query")
        return _normalize(np.array(r.embeddings[0]))


class HashingEmbedder:
    """N-gramas de caracteres (3-5) com hashing. Determinístico, sem rede. Não semântico — só para testes."""

    def __init__(self, dim: int = 512):
        self.name = f"hashing:{dim}"
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        t = unicodedata.normalize("NFKD", text.lower())
        t = "".join(c for c in t if not unicodedata.combining(c))
        v = np.zeros(self.dim, dtype=np.float32)
        for w in re.findall(r"\w+", t):
            w = f" {w} "
            for n in (3, 4, 5):
                for i in range(len(w) - n + 1):
                    h = int.from_bytes(hashlib.blake2b(w[i : i + n].encode(), digest_size=8).digest(), "little")
                    v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return _normalize(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))


def make_embedder(spec: str | None = None) -> Embedder:
    from dra.config import EMBEDDER

    spec = spec or os.environ.get("DRA_EMBEDDER") or EMBEDDER
    kind, _, arg = spec.partition(":")
    if kind == "fastembed":
        return FastEmbedEmbedder(arg)
    if kind == "voyage":
        return VoyageEmbedder(arg or "voyage-3.5")
    if kind == "hashing":
        return HashingEmbedder(int(arg) if arg else 512)
    raise ValueError(f"embedder desconhecido: {spec!r} (usa fastembed:<modelo>, voyage:<modelo> ou hashing)")
