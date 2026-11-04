"""Preço por milhão de tokens (USD), tarifas da API da Anthropic (tabela de 2026-10-06).

Escrita em cache = 1,25× a entrada (TTL de 5 min). Leitura de cache: valores publicados para Opus 5.5 e Sonnet 5.5
($0,20); para Haiku 5.5 assumo a mesma razão de 0,1× da entrada — confirmar em https://claude.com/pricing.
O pensamento (thinking) é faturado como saída e já vem incluído em `usage.output_tokens`.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES = {
    #                   entrada  saída  leitura cache
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    # destinos possíveis de fallback por recusa do classificador
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-opus-4-8": (5.00, 25.00, 0.50),
}


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        inp, out, cache_read = PRICES.get(self.model, PRICES["claude-opus-5-5"])
        return (
            self.input_tokens * inp
            + self.output_tokens * out
            + self.cache_read_tokens * cache_read
            + self.cache_write_tokens * inp * 1.25
        ) / 1_000_000

    def __add__(self, other: Usage) -> Usage:
        # usado para somar respostas do mesmo modelo (ex.: juiz); custo calculado com o modelo do primeiro
        return Usage(
            self.model,
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


def usage_from_response(resp) -> Usage:
    u = resp.usage
    return Usage(
        model=getattr(resp, "model", "") or "",
        input_tokens=u.input_tokens or 0,
        output_tokens=u.output_tokens or 0,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
    )
