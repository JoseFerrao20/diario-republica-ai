from __future__ import annotations

import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from dra import db
from dra.app import answerer, searcher
from dra.config import DEFAULT_DB

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Diário da República explicado")


def _db() -> str:
    return os.environ.get("DRA_DB", str(DEFAULT_DB))


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.post("/api/ask")
def ask(body: Ask) -> dict:
    ans_engine = answerer(_db())
    try:
        ans = ans_engine.answer(body.question)
    except anthropic.AuthenticationError as e:
        raise HTTPException(503, "Falta configurar ANTHROPIC_API_KEY no servidor.") from e
    except anthropic.RateLimitError as e:
        raise HTTPException(429, "Limite de pedidos da API atingido; tenta daqui a pouco.") from e
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Erro da API da Anthropic ({e.status_code}).") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(502, "Sem ligação à API da Anthropic.") from e
    run_id = db.log_run(ans_engine.searcher.conn, ans)
    return {"run_id": run_id, **ans.to_dict(), "stats": db.run_stats(ans_engine.searcher.conn)}


@app.get("/api/search")
def search(q: str, mode: str = "hybrid", k: int = 8) -> list[dict]:
    s = searcher(_db())
    out = []
    for h in s.search(q, k=k, mode=mode).hits:
        a = s.article(h.article_id)
        out.append({"key": a["key"], "diploma": a["diploma_name"], "article": a["number"],
                    "epigrafe": a["epigrafe"], "url": a["url"], "score": h.score, "ranks": h.ranks})
    return out


@app.get("/api/stats")
def stats() -> dict:
    return db.run_stats(searcher(_db()).conn)
