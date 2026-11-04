"""Resposta com fontes: pesquisa → artigos como documentos citáveis → Claude (Citations API) → resposta citada.

Cada artigo recuperado é um documento de conteúdo personalizado (`source.type = "content"`) cujos blocos são os
números/parágrafos do artigo. As citações devolvidas trazem `document_index` (→ artigo) e o intervalo de blocos
(→ números do artigo), e o `cited_text` é extraído pela API: não pode ser inventado pelo modelo.

Recusa em duas barreiras:
1. Antes do LLM (custo zero): nenhum artigo recuperado, ou a semelhança máxima abaixo do limiar.
2. Depois: o modelo responde "SEM_FONTE" quando os artigos não respondem; e uma resposta sem nenhuma citação é
   tratada como recusa (não mostramos afirmações jurídicas sem fonte).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import anthropic

from dra.config import ANSWER_EFFORT, ANSWER_MODEL
from dra.pricing import Usage, usage_from_response
from dra.search import Searcher

REFUSAL_MARK = "SEM_FONTE"
MAX_ARTICLE_CHARS = 12_000  # artigos-tabela enormes: só os primeiros parágrafos vão ao modelo

SYSTEM_PROMPT = f"""És um assistente que explica legislação portuguesa a cidadãos, em português de Portugal.

Recebes, em documentos, artigos de diplomas portugueses (texto consolidado do Diário da República). São a tua única
fonte: responde apenas com o que esses artigos dizem, e cita-os.

Como responder:
- Começa pela resposta direta (por exemplo, o prazo ou o número de dias), em linguagem simples, e depois explica as
  condições e exceções que os artigos preveem.
- Cada afirmação jurídica tem de estar apoiada numa citação de um dos documentos. Não acrescentes regras, prazos,
  valores ou exceções que não estejam nos documentos, mesmo que os conheças de outro lado: o texto pode ter mudado.
- Se os documentos só respondem a parte da pergunta, responde a essa parte e diz claramente o que fica por responder.
- Se nenhum documento responde à pergunta, ou a pergunta não é sobre legislação portuguesa, responde apenas com
  "{REFUSAL_MARK}:" seguido de uma frase a explicar o que procuraste e não encontraste. Não tentes responder.
- Não dês conselhos sobre o caso concreto da pessoa nem digas o que ela deve fazer; explica o que a lei diz.
- Sê conciso: no máximo 3 parágrafos curtos ou uma lista curta.
"""


@dataclass
class Citation:
    article_key: str
    diploma: str
    article: str          # "Artigo 238.º — Duração do período de férias"
    url: str
    cited_text: str
    blocks: tuple[int, int] | None  # números/parágrafos citados (0-indexados, fim exclusivo)


@dataclass
class AnswerSegment:
    text: str
    citations: list[int] = field(default_factory=list)  # índices em Answer.citations


@dataclass
class Answer:
    question: str
    refused: bool
    refusal_reason: str | None
    segments: list[AnswerSegment]
    citations: list[Citation]
    retrieved: list[dict]          # artigos enviados ao modelo (chave, diploma, artigo, url, ranks)
    usage: Usage | None
    retrieval_ms: float
    generation_ms: float
    stop_reason: str | None = None
    model: str | None = None

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd if self.usage else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(text=self.text, total_ms=self.total_ms, cost_usd=self.cost_usd)
        return d


def _document(art: dict) -> dict:
    paragraphs, size = [], 0
    for p in art["paragraphs"]:
        if size > MAX_ARTICLE_CHARS:
            paragraphs.append("[…] (artigo truncado; texto completo na ligação oficial)")
            break
        paragraphs.append(p)
        size += len(p)
    label = f"Artigo {art['number']}" + (f" — {art['epigrafe']}" if art["epigrafe"] else "")
    ctx = [f"Diploma: {art['diploma_title']}", f"Ligação oficial: {art['url']}"]
    if art["hierarchy"]:
        ctx.append(f"Localização: {art['hierarchy']}")
    if art["scope"] == "diploma" and art["diploma_name"].startswith("Código"):
        ctx.append("Nota: artigo do diploma que aprova o código, não do próprio código.")
    if art["last_amended"]:
        ctx.append(f"Última alteração do diploma: {art['last_amended']}")
    return {
        "type": "document",
        "source": {"type": "content", "content": [{"type": "text", "text": p} for p in paragraphs]},
        "title": f"{art['diploma_name']} — {label}",
        "context": "\n".join(ctx),
        "citations": {"enabled": True},
    }


class Answerer:
    def __init__(
        self,
        searcher: Searcher,
        client: anthropic.Anthropic | None = None,
        model: str = ANSWER_MODEL,
        effort: str = ANSWER_EFFORT,
        k: int = 8,
        min_similarity: float | None = None,
    ):
        self.searcher = searcher
        self.client = client
        self.model = model
        self.effort = effort
        self.k = k
        self.min_similarity = min_similarity

    def _client(self) -> anthropic.Anthropic:
        if self.client is None:
            self.client = anthropic.Anthropic()
        return self.client

    def answer(self, question: str) -> Answer:
        t0 = time.perf_counter()
        result = self.searcher.search(question, k=self.k)
        arts = [self.searcher.article(h.article_id) for h in result.hits]
        retrieved = [
            {"key": a["key"], "diploma": a["diploma_name"], "article": a["number"], "epigrafe": a["epigrafe"],
             "url": a["url"], "ranks": h.ranks, "similarity": h.similarity}
            for a, h in zip(arts, result.hits, strict=True)
        ]
        retrieval_ms = (time.perf_counter() - t0) * 1000

        reason = None
        if not arts:
            reason = "Nenhum artigo do corpus corresponde à pergunta."
        elif (
            self.min_similarity is not None
            and not result.explicit_refs
            and result.best_similarity is not None
            and result.best_similarity < self.min_similarity
        ):
            reason = "Os artigos encontrados não parecem relacionados com a pergunta."
        if reason:
            return Answer(question, True, reason, [], [], retrieved, None, retrieval_ms, 0.0)

        content = [_document(a) for a in arts] + [{"type": "text", "text": f"Pergunta: {question}"}]
        t1 = time.perf_counter()
        resp = self._client().beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": content}],
            thinking={"type": "adaptive"},
            output_config={"effort": self.effort},
            # Recusas dos classificadores de segurança (falsos positivos) são repetidas noutro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        generation_ms = (time.perf_counter() - t1) * 1000
        usage = usage_from_response(resp)
        return self._parse(question, resp, arts, retrieved, usage, retrieval_ms, generation_ms)

    def _parse(self, question, resp, arts, retrieved, usage, retrieval_ms, generation_ms) -> Answer:
        if resp.stop_reason == "refusal":
            return Answer(question, True, "O modelo recusou o pedido (classificador de segurança).", [], [],
                          retrieved, usage, retrieval_ms, generation_ms, resp.stop_reason, resp.model)
        citations: list[Citation] = []
        index: dict[tuple, int] = {}
        segments: list[AnswerSegment] = []
        for block in resp.content:
            if block.type != "text":
                continue
            seg = AnswerSegment(block.text)
            for c in getattr(block, "citations", None) or []:
                if getattr(c, "type", "") != "content_block_location":
                    continue
                a = arts[c.document_index]
                key = (a["key"], c.start_block_index, c.end_block_index)
                if key not in index:
                    index[key] = len(citations)
                    citations.append(Citation(
                        article_key=a["key"],
                        diploma=a["diploma_name"],
                        article=f"Artigo {a['number']}" + (f" — {a['epigrafe']}" if a["epigrafe"] else ""),
                        url=a["url"],
                        cited_text=c.cited_text,
                        blocks=(c.start_block_index, c.end_block_index),
                    ))
                seg.citations.append(index[key])
            segments.append(seg)

        text = "".join(s.text for s in segments).strip()
        if text.startswith(REFUSAL_MARK):
            reason = text[len(REFUSAL_MARK):].lstrip(" :").strip() or "Os artigos não respondem à pergunta."
            return Answer(question, True, reason, [], [], retrieved, usage, retrieval_ms, generation_ms,
                          resp.stop_reason, resp.model)
        if not citations:
            return Answer(question, True, "A resposta gerada não tinha nenhuma citação; não é mostrada.", [], [],
                          retrieved, usage, retrieval_ms, generation_ms, resp.stop_reason, resp.model)
        return Answer(question, False, None, segments, citations, retrieved, usage, retrieval_ms, generation_ms,
                      resp.stop_reason, resp.model)
