# Diário da República explicado

Perguntas sobre legislação portuguesa, respondidas com citações verificáveis: diploma, artigo e ligação para o
Diário da República. Quando as fontes não respondem, a aplicação diz que não sabe.

> [!WARNING]
> Projeto de portfólio em construção. Os textos usados são consolidados e **não têm valor legal**; as respostas não
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
        B[DRE screenservices<br/>adaptador direto] -.-> P
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
- **Serviços JSON internos do DRE** (`/dr/screenservices/...`, a aplicação OutSystems do site) — adaptador direto para
  refrescar e verificar diplomas.
- **RSS diário** (`files.diariodarepublica.pt/rss/serie1.xml`) — só o número do dia.

## Resultados da avaliação

_Por preencher: os números aparecem aqui depois de a avaliação correr no CI (fase 7 do plano)._

| Métrica | BM25 | Denso | Híbrido |
| --- | --- | --- | --- |
| Recall@5 | — | — | — |
| MRR | — | — | — |

| Métrica (resposta) | Valor |
| --- | --- |
| Fidelidade às fontes | — |
| Precisão das citações | — |
| Recusa correta | — |
| Custo médio por pergunta | — |
| Latência média (p95) | — |

## Como correr localmente

_Por preencher na fase 2._

## Decisões

Stack, alternativas e riscos: [`docs/PLANO.md`](docs/PLANO.md).
