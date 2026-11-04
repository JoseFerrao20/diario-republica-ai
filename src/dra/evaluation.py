"""Avaliação.

Recuperação (sem LLM, custo zero): Recall@k e MRR por modo (bm25, dense, hybrid).
Resposta (com LLM): recusa, precisão e cobertura das citações, suporte das citações e fidelidade (juiz),
correção face à referência (juiz), custo e latência.

O juiz é `claude-haiku-5-5` com saída estruturada (Pydantic). Só vê o texto dos artigos citados, não o corpus.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from dra.config import JUDGE_MODEL
from dra.pricing import Usage, usage_from_response
from dra.rag import Answer, Answerer
from dra.search import Searcher

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"


def load_questions(path: Path = EVAL_DIR / "perguntas.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- recuperação --------------------------------------------------------------------------------------------


def eval_retrieval(searcher: Searcher, questions: list[dict], modes=("bm25", "dense", "hybrid"), ks=(1, 3, 5, 8)):
    answerable = [q for q in questions if q["artigos"]]
    out: dict[str, dict] = {}
    per_q: dict[str, dict] = {}
    for mode in modes:
        if mode != "bm25" and searcher.embedder is None:
            continue
        recalls = {k: [] for k in ks}
        rr = []
        for q in answerable:
            hits = searcher.search(q["pergunta"], k=max(ks), mode=mode).hits
            keys = [searcher.article(h.article_id)["key"] for h in hits]
            expected = set(q["artigos"])
            for k in ks:
                recalls[k].append(len(expected & set(keys[:k])) / len(expected))
            first = next((i for i, kk in enumerate(keys, 1) if kk in expected), None)
            rr.append(1 / first if first else 0.0)
            per_q.setdefault(q["id"], {})[mode] = first
        out[mode] = {f"recall@{k}": round(statistics.mean(v), 3) for k, v in recalls.items()}
        out[mode]["mrr"] = round(statistics.mean(rr), 3)
        out[mode]["n"] = len(answerable)
    return {"summary": out, "first_relevant_rank": per_q}


# ---- juiz ---------------------------------------------------------------------------------------------------


class Claim(BaseModel):
    afirmacao: str
    suportada: bool


class FaithfulnessVerdict(BaseModel):
    afirmacoes: list[Claim]


class CorrectnessVerdict(BaseModel):
    veredicto: Literal["correta", "parcial", "errada"]
    justificacao: str


JUDGE_SYSTEM = "És um avaliador rigoroso de respostas sobre legislação portuguesa. Responde só no formato pedido."


class Judge:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = JUDGE_MODEL):
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.usage = Usage(model)

    def _parse(self, prompt: str, schema):
        resp = self.client.messages.parse(
            model=self.model,
            max_tokens=4000,
            system=JUDGE_SYSTEM,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
        )
        self.usage = self.usage + usage_from_response(resp)
        return resp.parsed_output

    def faithfulness(self, answer: str, sources: str) -> FaithfulnessVerdict:
        return self._parse(
            "Divide a RESPOSTA em afirmações factuais (regras, prazos, valores, condições). Para cada uma, diz se "
            "está suportada pelas FONTES: só conta como suportada se as fontes a afirmarem, não se for apenas "
            "plausível. Ignora frases de enquadramento sem conteúdo jurídico.\n\n"
            f"<fontes>\n{sources}\n</fontes>\n\n<resposta>\n{answer}\n</resposta>",
            FaithfulnessVerdict,
        )

    def correctness(self, question: str, answer: str, reference: str) -> CorrectnessVerdict:
        return self._parse(
            "Compara a RESPOSTA com a REFERÊNCIA para a PERGUNTA. 'correta': contém o essencial da referência sem "
            "contradições; 'parcial': falta parte importante ou é vaga; 'errada': contradiz a referência ou não "
            "responde.\n\n"
            f"<pergunta>{question}</pergunta>\n<referencia>{reference}</referencia>\n<resposta>{answer}</resposta>",
            CorrectnessVerdict,
        )


# ---- avaliação completa -------------------------------------------------------------------------------------


@dataclass
class QResult:
    id: str
    should_refuse: bool
    refused: bool
    cited: list[str]
    expected: list[str]
    citation_precision: float | None
    citation_recall: float | None
    faithfulness: float | None
    correctness: str | None
    cost_usd: float
    total_ms: float
    answer: str
    refusal_reason: str | None


def _sources_text(searcher: Searcher, answer: Answer) -> str:
    keys = sorted({c.article_key for c in answer.citations})
    parts = []
    for k in keys:
        r = searcher.conn.execute("SELECT id FROM articles WHERE key = ?", (k,)).fetchone()
        a = searcher.article(r["id"])
        parts.append(f"## {a['diploma_name']} — Artigo {a['number']}\n" + "\n".join(a["paragraphs"]))
    return "\n\n".join(parts)


def eval_answers(answerer: Answerer, judge: Judge | None, questions: list[dict], log=print) -> dict:
    results: list[QResult] = []
    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        ans = answerer.answer(q["pergunta"])
        should_refuse = bool(q.get("deve_recusar"))
        cited = sorted({c.article_key for c in ans.citations})
        expected = q["artigos"]
        prec = rec = faith = None
        corr = None
        if not ans.refused:
            if cited and expected:
                prec = len(set(cited) & set(expected)) / len(cited)
                rec = len(set(cited) & set(expected)) / len(expected)
            if judge:
                v = judge.faithfulness(ans.text, _sources_text(answerer.searcher, ans))
                faith = (sum(c.suportada for c in v.afirmacoes) / len(v.afirmacoes)) if v.afirmacoes else 1.0
                if not should_refuse:
                    corr = judge.correctness(q["pergunta"], ans.text, q["referencia"]).veredicto
        results.append(QResult(q["id"], should_refuse, ans.refused, cited, expected, prec, rec, faith, corr,
                               ans.cost_usd, ans.total_ms, ans.text, ans.refusal_reason))
        log(f"[{i}/{len(questions)}] {q['id']}: {'recusou' if ans.refused else 'respondeu'} "
            f"cit={cited} ${ans.cost_usd:.4f} {(time.perf_counter() - t0):.1f}s")
    return {"summary": summarize(results, judge), "results": [r.__dict__ for r in results]}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else None


def summarize(results: list[QResult], judge: Judge | None) -> dict:
    ans = [r for r in results if not r.should_refuse]
    neg = [r for r in results if r.should_refuse]
    answered = [r for r in ans if not r.refused]
    costs = [r.cost_usd for r in results if r.cost_usd > 0]
    lat = sorted(r.total_ms for r in results)
    corr = [r.correctness for r in answered if r.correctness]
    return {
        "n": len(results),
        "taxa_resposta": round(len(answered) / len(ans), 3) if ans else None,          # respondeu quando devia
        "recusa_correta": round(sum(r.refused for r in neg) / len(neg), 3) if neg else None,
        "precisao_citacoes": _mean(r.citation_precision for r in answered),
        "cobertura_citacoes": _mean(r.citation_recall for r in answered),
        "fidelidade": _mean(r.faithfulness for r in results if not r.refused),
        "correcao": {k: corr.count(k) for k in ("correta", "parcial", "errada")} if corr else None,
        "custo_medio_usd": round(statistics.mean(costs), 5) if costs else 0.0,
        "custo_medio_todas_usd": round(statistics.mean(r.cost_usd for r in results), 5) if results else 0.0,
        "latencia_media_ms": round(statistics.mean(lat)) if lat else None,
        "latencia_p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))]) if lat else None,
        "custo_juiz_usd": round(judge.usage.cost_usd, 4) if judge else 0.0,
    }


def check_thresholds(summary: dict, thresholds: dict) -> list[str]:
    """Devolve as métricas abaixo do limiar (vazio = passou)."""
    fails = []
    for path, minimum in thresholds.items():
        cur = summary
        for p in path.split("."):
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None or cur < minimum:
            fails.append(f"{path} = {cur} < {minimum}")
    return fails
