"""dra — linha de comandos.

    dra ingest [--embedder fastembed:<modelo>|voyage:<modelo>|hashing|none] [--mirror-url URL]
    dra search "pergunta" [--mode hybrid|bm25|dense]
    dra ask "pergunta"
    dra serve [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dra import db
from dra.config import DATA_DIR, DEFAULT_DB, load_corpus


def cmd_ingest(a: argparse.Namespace) -> None:
    from dra import index
    from dra.embed import make_embedder
    from dra.sources import mirror

    corpus = load_corpus()
    root = mirror.fetch(corpus, Path(a.mirror_dir), repo_url=a.mirror_url)
    print(f"espelho em {root} (commit {corpus.mirror_commit[:10]})")
    emb = None if a.embedder == "none" else make_embedder(a.embedder)
    if emb:
        print(f"embeddings: {emb.name} ({emb.dim} dimensões)")
    conn = db.connect(a.db)
    print(json.dumps(index.build(conn, corpus, mirror.iter_articles(corpus, root), emb), ensure_ascii=False))


def cmd_search(a: argparse.Namespace) -> None:
    from dra.app import searcher

    s = searcher(a.db)
    r = s.search(a.question, k=a.k, mode=a.mode)
    for h in r.hits:
        art = s.article(h.article_id)
        sim = f" sim={h.similarity:.3f}" if h.similarity is not None else ""
        print(f"{h.score:.4f} {art['diploma_name']} — Artigo {art['number']} {art['epigrafe'] or ''} {h.ranks}{sim}")


def cmd_ask(a: argparse.Namespace) -> None:
    from dra.app import answerer

    ans = answerer(a.db).answer(a.question)
    db.log_run(answerer(a.db).searcher.conn, ans)
    if ans.refused:
        print(f"Sem resposta: {ans.refusal_reason}")
    else:
        for seg in ans.segments:
            print(seg.text + "".join(f"[{i + 1}]" for i in seg.citations), end="")
        print("\n")
        for i, c in enumerate(ans.citations, 1):
            print(f"[{i}] {c.diploma}, {c.article} — {c.url}\n    «{c.cited_text.strip()[:200]}»")
    if ans.usage:
        u = ans.usage
        print(f"\n{u.model}: {u.input_tokens} tokens entrada ({u.cache_read_tokens} cache), {u.output_tokens} saída, "
              f"${ans.cost_usd:.4f}, pesquisa {ans.retrieval_ms:.0f} ms, geração {ans.generation_ms:.0f} ms")


def cmd_serve(a: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["DRA_DB"] = a.db
    uvicorn.run("dra.api:app", host=a.host, port=a.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="dra", description="Diário da República explicado")
    p.add_argument("--db", default=str(DEFAULT_DB))
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="descarrega o espelho e constrói o índice")
    s.add_argument("--embedder", default=None, help="fastembed:<modelo> | voyage:<modelo> | hashing | none")
    s.add_argument("--mirror-dir", default=str(DATA_DIR / "mirror"))
    s.add_argument("--mirror-url", default=None, help="outro remoto git (ex.: file:// de uma cópia local)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="só a pesquisa, sem LLM")
    s.add_argument("question")
    s.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    s.add_argument("-k", type=int, default=8)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="pergunta com resposta citada (precisa de ANTHROPIC_API_KEY)")
    s.add_argument("question")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("serve", help="interface web")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    sys.exit(main())
