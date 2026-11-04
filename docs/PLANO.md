# Plano técnico

Estado: proposta inicial (primeiro commit). Este documento regista as decisões e o porquê de cada uma.

## 1. Fontes de dados (verificadas, não inventadas)

Investigação feita a 2026-11-04. O Diário da República Eletrónico (DRE, publicado pela INCM) não tem uma API pública
documentada. Há três vias reais de acesso, todas confirmadas no código de projetos públicos que as usam hoje:

| Fonte | O que dá | Limites | Onde a vi |
| --- | --- | --- | --- |
| `https://files.diariodarepublica.pt/rss/serie1.xml` (e `serie2.xml`, `serie1-html.xml`) | Atos publicados **no próprio dia**: título, sumário, PDF | Só o número do dia; não há histórico | `diogogs/dr-watch` (`src/ingestion/rss.py`), `gonmsilva/dr-resumido` (`dr_digest/feeds.py`) |
| `POST https://diariodarepublica.pt/dr/screenservices/dr/<Módulo>/<Ecrã>/<Ação>` | Os serviços JSON internos (OutSystems) que o site usa: pesquisa, detalhe de diploma, legislação consolidada por fragmento (artigo) | Não documentados; precisam do `versionToken` de `/dr/moduleservices/moduleinfo` e do `apiVersion` lido do JS do ecrã; podem mudar sem aviso; exigem ritmo baixo (≤ 2 pedidos/s) | `LawAgents/pt` (`scripts/dre_client.py`), `nikuscs/lauyer` (`src/dr/detail.rs`, `src/dr/search.rs`) |
| `https://github.com/LawAgents/pt` | Espelho em markdown de ~4.987 diplomas **consolidados** em vigor, extraídos do DRE com os serviços acima; cabeçalho YAML com `dre_id`, tipo, número, data, emissor, `source` (ligação oficial) e `original_pdf`; um título `Artigo N.º — Epígrafe` por artigo | Atualização semanal; é uma terceira parte (scripts MIT; os textos legais não têm licença própria — art. 8.º do CDADC exclui os textos oficiais da proteção) | clonado e inspecionado |

Ações concretas dos serviços internos (do `dre_client.py` do LawAgents/pt):

- `LegislacaoConsolidada/LegCons_Detalhe/DataActionGetData` — diploma consolidado (`DiplomaLegisId`)
- `LegislacaoConsolidada/LegCons_Detalhe/DataActionGetDiplomaFragByIdAndApplicationSetting` — um fragmento (artigo)
- `Legislacao_Conteudos/Conteudo_Detalhe/DataActionGetAllConteudoDetalheData` — texto original
- `Pesquisas/PesquisaResultado/DataActionGetPesquisas` — pesquisa
- `MenusParaPesquisa/LegislacaoPorCodigo/DataActionGetCodigos` — lista de códigos

O que **não** existe: não há dataset do DRE em dados.gov.pt (procurei "diário da república", "legislação" e a
organização INCM — os resultados são avisos municipais e planos de ordenamento), nem dumps oficiais.

### Decisão

1. **Fonte principal para o índice: espelho LawAgents/pt** (fixado num commit concreto, para ser reprodutível).
   Já vem dividido por artigo, com metadados e ligação oficial. Cada citação aponta para a ligação do DRE
   (`source`), nunca para o espelho.
2. **Adaptador direto ao DRE** (`screenservices`), para refrescar diplomas e verificar texto na fonte oficial quando
   se corre localmente. Fica atrás de uma interface `Source`, com limitação de ritmo, User-Agent identificado e paragem
   imediata em 429/503. Nota honesta: o ambiente onde este repositório foi criado não chega a `diariodarepublica.pt`
   (bloqueado pela política de rede), por isso este adaptador só pode ser validado fora dele.
3. **RSS diário** para uma funcionalidade secundária ("o que saiu hoje"), fora do âmbito do MVP.

### Corpus do MVP

Não indexo os 4.987 diplomas na primeira versão: o conjunto de avaliação tem de ser verificável à mão. Corpus inicial
(~20 diplomas muito consultados, alguns milhares de artigos):

Constituição; Código do Trabalho; Código Civil; Código do Procedimento Administrativo; Código da Estrada;
Código do IRS; Código do IVA; Lei 58/2019 (execução do RGPD); Lei 24/96 (defesa do consumidor);
DL 84/2021 (garantias na compra e venda de bens de consumo); DL 446/85 (cláusulas contratuais gerais);
NRAU (Lei 6/2006); Código das Sociedades Comerciais; Lei Geral do Trabalho em Funções Públicas (Lei 35/2014);
Código de Processo Civil; Código Penal; Código da Insolvência; Lei da Nacionalidade (Lei 37/81).

A lista exata fica em `config/corpus.yaml`; alargar é mudar a lista e reindexar.

## 2. Stack

| Camada | Escolha | Porquê | Alternativa considerada |
| --- | --- | --- | --- |
| Linguagem | Python 3.12+, `uv` | Ecossistema de RAG/avaliação; SDK oficial da Anthropic | TypeScript |
| Base de dados | **SQLite** com **FTS5** (BM25) + **sqlite-vec** (vetores) | Zero infraestrutura: o índice é um ficheiro, corre igual no portátil e no CI; FTS5 e sqlite-vec chegam para dezenas de milhares de artigos | PostgreSQL + pgvector + `tsvector` — melhor para produção multiutilizador; fica como passo seguinte documentado |
| Pesquisa híbrida | BM25 (FTS5, `unicode61 remove_diacritics 2`) + kNN denso, fundidos por **Reciprocal Rank Fusion**; reforço por referência explícita ("artigo 238.º do Código do Trabalho" → filtro direto) | RRF não precisa de calibrar escalas entre BM25 e cosseno; as perguntas jurídicas citam muitas vezes artigos pelo número | Combinação linear de pontuações |
| Embeddings | Interface `Embedder` com: `fastembed` local (`intfloat/multilingual-e5-large` ou `paraphrase-multilingual-MiniLM-L12-v2` para o CI) e **Voyage AI** (parceiro recomendado pela Anthropic, que não tem API de embeddings própria) | Português exige modelo multilingue; local = custo zero e reprodutível no CI | OpenAI embeddings (fora: o projeto é Anthropic-only) |
| Divisão | **Um chunk por artigo** (com epígrafe e caminho hierárquico: Livro › Título › Capítulo); artigos longos partidos por números (`1 -`, `2 -`) mantendo o mesmo identificador de artigo | A citação natural em direito é o artigo; o chunk é a unidade citável | Janelas de N tokens (partem artigos ao meio) |
| LLM | API da Anthropic, SDK `anthropic` para Python, **`claude-opus-5-5`** para responder (pensamento adaptativo, `effort` explícito); **`claude-haiku-5-5`** como juiz na avaliação | Modelos mais recentes; o juiz barato mantém o custo do CI baixo | — |
| Citações | **Citations API nativa**: cada artigo recuperado entra como bloco `document` com `citations: {enabled: true}`; a resposta traz `cited_text` e o índice do documento, que mapeio para diploma + artigo + ligação do DRE | As citações são verificáveis por construção (o texto citado tem de existir na fonte); não dependem de o modelo escrever ids corretamente | Pedir ids `[1]` no texto e validar depois |
| Recusa | Duas barreiras: (1) sem resultados acima de um limiar → recusa sem chamar o LLM (custo zero); (2) instrução de sistema para recusar quando os artigos não respondem, e uma resposta sem nenhuma citação é tratada como recusa | Não inventar direito | — |
| API + interface | **FastAPI** + página HTML simples (sem framework JS) com resposta, citações clicáveis, tokens, custo em € /$ e latência por fase | Leve; a interface mostra o que interessa avaliar | Streamlit |
| Avaliação | Conjunto próprio `eval/perguntas.jsonl` (pergunta, resposta de referência, artigos esperados, `deve_recusar`); métricas abaixo; corre no **GitHub Actions** | — | RAGAS (pesado e opaco; prefiro métricas explícitas) |
| Observabilidade | Cada resposta guarda tokens de entrada/saída/cache, custo calculado com a tabela de preços, latência de recuperação e de geração, em SQLite (`runs`) | Requisito do projeto; dá o custo médio por pergunta para o README | OpenTelemetry (fica para depois) |

## 3. Avaliação

Conjunto de teste: ~60 perguntas escritas à mão sobre o corpus, cada uma com os artigos que a respondem e uma resposta
de referência curta; ~15% são perguntas **sem resposta no corpus** (devem ser recusadas).

| Métrica | Como se mede | Precisa de LLM? |
| --- | --- | --- |
| Recall@k e MRR da recuperação | artigos esperados vs. top-k (BM25, denso e híbrido, para comparar) | Não |
| Precisão das citações | fração das citações da resposta que apontam para um artigo esperado ou que o juiz considera suporte válido | Parcial |
| Cobertura das citações | fração dos artigos esperados citados | Não |
| Fidelidade às fontes | juiz (`claude-haiku-5-5`) divide a resposta em afirmações e verifica cada uma contra os artigos citados; fração suportada | Sim |
| Correção | juiz compara com a resposta de referência (correta / parcial / errada) | Sim |
| Recusa | precisão e recall da recusa nas perguntas sem resposta | Não |
| Custo e latência | média e p95 por pergunta | Não |

No CI: a avaliação de recuperação corre em todos os PRs (sem custo); a avaliação completa corre quando o segredo
`ANTHROPIC_API_KEY` existe e falha se as métricas descerem abaixo de limiares guardados em `eval/limiares.yaml`.

## 4. Fases

1. **Fundações** — este commit: README, plano, estrutura.
2. **Ingestão** — leitor do espelho (YAML + divisão por artigo), adaptador DRE, esquema SQLite, `dra ingest`.
3. **Pesquisa** — BM25, denso, RRF, deteção de referências explícitas; testes unitários.
4. **Avaliação da recuperação** — conjunto de perguntas, métricas sem LLM, CI.
5. **RAG** — geração com a Citations API, recusa, registo de tokens/custo/latência.
6. **Interface** — FastAPI + página.
7. **Avaliação completa** — juiz LLM, limiares, resultados no README.

## 5. Riscos

- **Serviços internos do DRE mudam** sem aviso (OutSystems faz redeploys) → o espelho é a fonte do índice; o
  adaptador direto deteta mudanças de versão e falha de forma explícita.
- **Texto consolidado não tem valor legal** → aviso permanente na interface e em cada resposta.
- **Dependência de terceiros (LawAgents/pt)** → commit fixado; o formato é simples e o adaptador direto é o plano B.
- **Avaliação enviesada por eu escrever as perguntas** → perguntas escritas a partir de dúvidas reais típicas (trabalho,
  consumo, arrendamento), não a partir do texto dos artigos; resultados de BM25 sozinho publicados como linha de base.
