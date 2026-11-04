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

Medido no GitHub Actions sobre o índice completo (4.979 artigos), nas 51 perguntas com resposta. Recall@k = fração
dos artigos esperados que aparecem nos k primeiros; MRR = média de 1/posição do primeiro artigo certo.

| Modo | Recall@1 | Recall@3 | Recall@5 | Recall@8 | MRR |
| --- | --- | --- | --- | --- | --- |
| BM25 (FTS5) | 0,33 | 0,59 | 0,69 | 0,73 | 0,47 |
| Denso — paraphrase-multilingual-MiniLM-L12-v2 | 0,45 | 0,68 | 0,76 | 0,79 | 0,58 |
| **Híbrido BM25 + MiniLM (RRF)** | **0,53** | **0,72** | **0,78** | **0,82** | **0,64** |
| Denso — multilingual-e5-large | pendente (CI) | | | | |
| Híbrido BM25 + e5-large (RRF) | pendente (CI) | | | | |

O híbrido supera os dois componentes: o BM25 falha no vocabulário ("despedir-me" vs. "denúncia", "garantia" vs.
"falta de conformidade") e o denso falha em termos exatos e números; a fusão recupera o melhor de cada um. O modelo
de resposta recebe os 8 primeiros artigos, por isso Recall@8 é o teto do que pode citar corretamente.

### Respostas (com LLM)

**Não medido.** A avaliação das respostas precisa de uma chave da API da Anthropic, que é paga, e este projeto foi
construído sem gastos. O código está pronto e testado com um cliente simulado (`tests/test_rag.py`); quem tiver uma
chave corre `uv run dra eval` e obtém as métricas abaixo. Não publico números que não medi.

| Métrica | Como se mede |
| --- | --- |
| Recusa correta | % das 8 perguntas sem fonte que foram recusadas |
| Taxa de resposta | % das 51 perguntas com fonte que foram respondidas |
| Precisão das citações | artigos citados que estão entre os esperados |
| Cobertura das citações | artigos esperados que foram citados |
| Fidelidade | afirmações da resposta suportadas pelos artigos citados (juiz `claude-haiku-5-5`) |
| Correção | correta / parcial / errada face à referência (juiz) |
| Custo e latência | média e p95 por pergunta, a partir de `usage` de cada resposta |

### Custo por pergunta (estimativa, não medição)

Medido sem chamar a API: o contexto enviado ao modelo (prompt de sistema + 8 artigos + pergunta) tem em média
**~16.900 caracteres** por pergunta do conjunto de teste (mediana 15.400; p95 34.000).

| | Pressuposto | Opus 5.5 ($4 / $20 por M) |
| --- | --- | --- |
| Entrada | 3–4 caracteres por token → ~4.200–5.600 tokens | ~$0,017–0,022 |
| Saída | resposta curta + pensamento adaptativo (`effort: medium`): 300–1.500 tokens | ~$0,006–0,030 |
| **Total** | | **~$0,02–0,05 por pergunta** |

O prompt de sistema vai em cache, mas é pequeno; o grosso são os artigos, que mudam a cada pergunta. A avaliação
completa (59 perguntas + juiz) custaria assim cerca de US$ 1,5–3. Trocar o modelo de resposta por
`claude-haiku-5-5` (`DRA_ANSWER_MODEL`) baixa o custo cerca de 40 vezes, à custa de qualidade por medir.
Na aplicação, o custo real de cada resposta é calculado a partir de `usage` e mostrado na página.

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
mensagem contenha `[eval]`, e só se o segredo `ANTHROPIC_API_KEY` estiver definido no repositório; sem ele, o job
termina sem fazer nada e sem custos.

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
