# Diário da República explicado

Perguntas sobre legislação portuguesa, respondidas com citações verificáveis: diploma, artigo e ligação para o
Diário da República. Quando as fontes não respondem, a aplicação diz que não sabe.

> [!WARNING]
> Projeto de portfólio. Os textos usados são consolidados e **não têm valor legal**; as respostas não
> substituem aconselhamento jurídico. Confirma sempre na ligação oficial.

## O problema

A legislação portuguesa é pública, mas difícil de usar: um diploma é alterado dezenas de vezes, a versão em vigor está
espalhada, e a pesquisa do DRE é por palavras. Um LLM sozinho responde com confiança e inventa artigos. Este projeto
junta as duas coisas da forma certa: **pesquisa híbrida sobre o texto oficial, dividido por artigo**, e um modelo
Claude que só responde com base no que recuperou, **citando o texto exato**.

## Arquitetura

```mermaid
flowchart LR
    subgraph Ingestão
        A[LawAgents/pt<br/>espelho do DRE consolidado] --> P[Parser<br/>YAML + 1 chunk por artigo]
        B[DRE screenservices<br/>adaptador direto — planeado] -.-> P
        P --> E[Embeddings<br/>multilingue]
        P --> DB[(SQLite<br/>FTS5 BM25 + sqlite-vec)]
        E --> DB
    end
    subgraph Pergunta
        Q[Pergunta] --> R[Pesquisa híbrida<br/>BM25 + kNN + RRF<br/>+ referências explícitas]
        DB --> R
        R -->|nada relevante| X[Recusa sem chamar o LLM]
        R -->|top-k artigos| L[Claude Opus 5.5<br/>Citations API]
        L --> C[Resposta + citações<br/>diploma · artigo · ligação DRE]
        L --> M[(Registo: tokens, custo,<br/>latência por fase)]
    end
    subgraph Avaliação no CI
        T[eval/perguntas.jsonl] --> V[Recall@k · MRR · precisão das citações<br/>fidelidade · recusa · custo]
    end
```

## Fontes de dados

O DRE não tem API pública documentada. As três vias reais (verificadas, ver [`docs/PLANO.md`](docs/PLANO.md)):

- **[LawAgents/pt](https://github.com/LawAgents/pt)** — espelho em markdown de ~5.000 diplomas consolidados, extraídos do
  DRE, com `dre_id` e ligação oficial em cada ficheiro. É a fonte do índice.
- **Serviços JSON internos do DRE** (`/dr/screenservices/...`, a aplicação OutSystems do site) — a mesma via que o
  espelho usa. Um adaptador direto está planeado mas **ainda não implementado**: o ambiente onde o projeto foi
  construído não chega a `diariodarepublica.pt`, e não publico um cliente que não pude testar.
- **RSS diário** (`files.diariodarepublica.pt/rss/serie1.xml`) — só o número do dia.

## Resultados da avaliação

Conjunto de teste: [`eval/perguntas.jsonl`](eval/perguntas.jsonl) — 59 perguntas escritas à mão em linguagem
corrente (trabalho, consumo, arrendamento, Constituição, impostos, procedimento administrativo, dados pessoais,
nacionalidade), cada uma com os artigos que a respondem e uma resposta de referência tirada do texto; 8 perguntas
não têm resposta no corpus e **devem ser recusadas** (incluindo uma armadilha: o art. 27.º do Código da Estrada está
indexado, mas a tabela de limites de velocidade não foi extraída do DRE).

### Recuperação (sem LLM, corre em todos os pushes)

_Os números com modelos de embeddings reais são produzidos pelo CI (`retrieval-eval`) e entram aqui depois da
primeira execução._ Linha de base medida localmente (4.979 artigos, 51 perguntas com resposta):

| Modo | Recall@1 | Recall@5 | Recall@8 | MRR |
| --- | --- | --- | --- | --- |
| BM25 (FTS5) | 0,33 | 0,67 | 0,73 | 0,47 |
| n-gramas com hashing (controlo, **não semântico**) | 0,31 | 0,57 | 0,61 | 0,41 |
| Híbrido BM25 + hashing | 0,39 | 0,65 | 0,70 | 0,53 |
| Denso — multilingual-MiniLM-L12 | CI | CI | CI | CI |
| Denso — multilingual-e5-large | CI | CI | CI | CI |
| Híbrido BM25 + e5-large | CI | CI | CI | CI |

O que falha no BM25 é vocabulário: "despedir-me" vs. "denúncia", "garantia" vs. "falta de conformidade",
"morrer o meu pai" vs. "falecimento de parente no 1.º grau". É o caso para a pesquisa semântica.

### Respostas (com LLM)

_Por medir: precisa de `ANTHROPIC_API_KEY` (ver "Avaliação" abaixo)._

| Métrica | Como se mede | Valor |
| --- | --- | --- |
| Recusa correta | % das 8 perguntas sem fonte que foram recusadas | — |
| Taxa de resposta | % das 51 perguntas com fonte que foram respondidas | — |
| Precisão das citações | artigos citados que estão entre os esperados | — |
| Cobertura das citações | artigos esperados que foram citados | — |
| Fidelidade | afirmações da resposta suportadas pelos artigos citados (juiz `claude-haiku-5-5`) | — |
| Correção | correta / parcial / errada face à referência (juiz) | — |
| Custo médio por pergunta | tokens × tarifa (inclui pensamento e cache) | — |
| Latência média / p95 | pesquisa + geração | — |

## Como correr localmente

Requisitos: Python 3.12+, [uv](https://docs.astral.sh/uv/), git.

```bash
git clone https://github.com/JoseFerrao20/diario-republica-ai && cd diario-republica-ai
uv sync --extra local-embeddings

# 1. Ingestão: descarrega só os ~14 diplomas do corpus do espelho (commit fixado) e constrói o índice.
#    O modelo de embeddings (~2 GB para o e5-large) é descarregado na primeira vez.
uv run dra ingest --embedder fastembed:intfloat/multilingual-e5-large
#    Mais leve: --embedder fastembed:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#    Só BM25, sem descarregar modelos: --embedder none

# 2. Pesquisa (sem LLM)
uv run dra search "quantos dias de férias tenho por ano"

# 3. Pergunta com resposta citada
export ANTHROPIC_API_KEY=...
uv run dra ask "Com quanta antecedência tenho de avisar para me despedir?"

# 4. Interface web em http://127.0.0.1:8000
uv run dra serve
```

Variáveis úteis: `DRA_ANSWER_MODEL` (por omissão `claude-opus-5-5`), `DRA_ANSWER_EFFORT` (`medium`),
`DRA_JUDGE_MODEL` (`claude-haiku-5-5`), `DRA_DB`.

### Avaliação

```bash
uv run pytest                     # 22 testes, sem rede nem chave
uv run dra eval-retrieval         # Recall@k e MRR por modo; --check falha abaixo de eval/limiares.yaml
uv run dra eval                   # completa: respostas + juiz (~59 chamadas ao Opus 5.5 + ~100 ao Haiku 5.5)
```

No GitHub Actions, a avaliação da recuperação corre em todos os pushes e PRs, com dois modelos de embeddings.
A avaliação completa corre manualmente (_Actions → CI → Run workflow → full_eval_) ou num push para `main` cuja
mensagem contenha `[eval]`, e só se o segredo `ANTHROPIC_API_KEY` estiver definido no repositório.

## Estrutura

```
config/corpus.yaml        diplomas indexados e commit fixado do espelho
src/dra/parse.py          markdown do espelho → artigos (hierarquia, anexos, notas de alteração)
src/dra/sources/mirror.py descarga (sparse checkout) e ordem das partes dos códigos
src/dra/index.py          SQLite: artigos, FTS5, chunks, vetores
src/dra/search.py         BM25, kNN, RRF, referências explícitas
src/dra/rag.py            Claude + Citations API, recusa
src/dra/pricing.py        custo por resposta
src/dra/evaluation.py     métricas e juiz
src/dra/api.py, static/   FastAPI + página
eval/                     perguntas, limiares, resultados
```

## Decisões

Stack, alternativas e riscos: [`docs/PLANO.md`](docs/PLANO.md).
