"""Adaptador direto ao Diário da República Eletrónico (diariodarepublica.pt).

O DRE não tem API pública documentada. O site é uma aplicação OutSystems e os dados vêm de serviços JSON internos
("screen services"): POST /dr/screenservices/dr/<Módulo>/<Ecrã>/<Ação>. Cada pedido leva o `versionToken` do módulo
(GET /dr/moduleservices/moduleinfo) e o `apiVersion` da ação, que só aparece no JavaScript do ecrã. Estes detalhes
vêm do cliente do projeto LawAgents/pt (scripts/dre_client.py, licença MIT), que os usa semanalmente; o formato da
resposta foi validado contra respostas reais gravadas por esse projeto (tests/fixtures/dre/).

Regras de cortesia (os serviços não são públicos nem documentados):
- no máximo ~2 pedidos por segundo, com User-Agent que identifica o projeto;
- paragem imediata em HTTP 429/503 ou após 3 erros seguidos;
- se o formato da resposta mudar, falha com `ServiceChanged` em vez de adivinhar.

Uso: `dra dre-sync <diploma>` compara o texto indexado com o DRE e, com `--apply`, substitui-o.
"""

from __future__ import annotations

import http.client
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from dra.parse import Article, annex_scope

SITE = "https://diariodarepublica.pt"
BASE = SITE + "/dr/"
USER_AGENT = "diario-republica-ai/0.1 (+https://github.com/JoseFerrao20/diario-republica-ai)"
ANON_CSRF = "T6C+9iB49TLra4jEsMeSckDMNhQ="  # token CSRF anónimo do OutSystems, igual para todos os visitantes
MIN_INTERVAL = 0.55
MAX_CONSECUTIVE_ERRORS = 3
SLOW_DOWN = (429, 503)

Transport = Callable[[str, bytes | None, dict], tuple[int, bytes]]


class FetchAborted(Exception):
    """Demasiados erros, ou o servidor pediu para abrandar: parar tudo."""


class ServiceChanged(FetchAborted):
    """O serviço respondeu num formato desconhecido: o site mudou."""


def urllib_transport(url: str, data: bytes | None, headers: dict, timeout: float = 120) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""
    except urllib.error.URLError as e:
        raise OSError(str(e.reason)) from e
    except http.client.HTTPException as e:
        raise OSError(f"{type(e).__name__}: {e}") from e


class DREClient:
    def __init__(
        self,
        transport: Transport = urllib_transport,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        min_interval: float = MIN_INTERVAL,
        log: Callable[[str], None] | None = None,
    ):
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self.min_interval = min_interval
        self._log = log or (lambda m: print(m, file=sys.stderr))
        self._next_start = clock()
        self._lock = threading.Lock()
        self.errors_in_row = 0
        self.requests = 0
        self.version: str | None = None
        self._urls: dict[str, str] = {}
        self._api: dict[tuple[str, str, str], str] = {}

    # ---- HTTP com ritmo e contagem de erros -------------------------------------------------------------

    def _request(self, url: str, data: bytes | None = None, headers: dict | None = None) -> bytes:
        h = {"User-Agent": USER_AGENT, **(headers or {})}
        while True:
            with self._lock:
                now = self._clock()
                start = max(now, self._next_start)
                self._next_start = start + self.min_interval
                self.requests += 1
            if start > now:
                self._sleep(start - now)
            try:
                status, body = self._transport(url, data, h)
                err = None if status == 200 else f"HTTP {status}"
            except OSError as e:
                status, body, err = None, b"", f"erro de rede: {e}"
            if err is None:
                self.errors_in_row = 0
                return body
            self.errors_in_row += 1
            self._log(f"pedido falhou ({err}): {url}")
            if status in SLOW_DOWN:
                raise FetchAborted(f"{err}: o servidor pediu para abrandar ({url})")
            if self.errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                raise FetchAborted(f"{self.errors_in_row} erros seguidos; último: {err} ({url})")

    def _json(self, body: bytes, what: str) -> dict:
        try:
            return json.loads(body)
        except ValueError as e:
            raise ServiceChanged(f"{what}: a resposta não é JSON") from e

    # ---- serviços OutSystems ----------------------------------------------------------------------------

    def refresh(self) -> None:
        mi = self._json(self._request(BASE + "moduleservices/moduleinfo"), "moduleinfo")
        try:
            self.version = mi["manifest"]["versionToken"]
            self._urls = mi["manifest"]["urlVersions"]
        except (KeyError, TypeError) as e:
            raise ServiceChanged("moduleinfo com formato desconhecido") from e
        self._api = {}

    def api_version(self, module: str, screen: str, action: str) -> str:
        key = (module, screen, action)
        if key not in self._api:
            path = f"/dr/scripts/dr.{module}.{screen}.mvc.js"
            js = self._request(SITE + path + self._urls.get(path, "")).decode("utf-8", "replace")
            m = re.search(rf'callDataAction\("{re.escape(action)}", "[^"]*", "[^"]*", "([^"]*)"', js)
            if not m:
                raise ServiceChanged(f"apiVersion de {action} não encontrado em {path}")
            self._api[key] = m.group(1)
        return self._api[key]

    def data_action(self, module: str, screen: str, action: str, variables: dict) -> dict:
        if self.version is None:
            self.refresh()
        url = f"{BASE}screenservices/dr/{module}/{screen}/{action}"
        for _ in range(3):
            body = {
                "versionInfo": {"moduleVersion": self.version, "apiVersion": self.api_version(module, screen, action)},
                "viewName": f"{module}.{screen}",
                "screenData": {"variables": variables},
                "clientVariables": {},
            }
            j = self._json(
                self._request(url, json.dumps(body).encode(), {
                    "X-CSRFToken": ANON_CSRF, "Accept": "application/json",
                    "Content-Type": "application/json; charset=UTF-8",
                }),
                action,
            )
            vi = j.get("versionInfo") or {}
            if vi.get("hasModuleVersionChanged") or vi.get("hasApiVersionChanged"):
                self._log("o DRE foi atualizado; a renovar os tokens de versão")
                self.refresh()
                continue
            if "exception" in j or "data" not in j:
                self.errors_in_row += 1
                if self.errors_in_row >= MAX_CONSECUTIVE_ERRORS:
                    raise FetchAborted(f"{action}: {str(j.get('exception'))[:200]}")
                continue
            return j["data"]
        raise ServiceChanged(f"{action}: sem resposta utilizável após 3 tentativas")

    def consolidated(self, diploma_legis_id: str | int) -> dict:
        """Texto consolidado de um diploma (todos os fragmentos: capítulos, artigos, anexos)."""
        return self.data_action(
            "LegislacaoConsolidada", "LegCons_Detalhe", "DataActionGetData",
            {"DiplomaLegisId": str(diploma_legis_id), "DiplomaFragId": "0", "GetDiplomaFragByIdAndApplicationSetting": {}},
        )


# ---- conversão da resposta em artigos -------------------------------------------------------------------

ESTADO_REVOGADO = "4"  # VersaoEstadoId observado nos fragmentos revogados (6 = em vigor)
TIPO_ARTIGO = "11"     # TipoFragmentoId de um artigo (1 capítulo, 8 secção, 14 anexo, 15 raiz, 7 assinatura)
_ART_RE = re.compile(r"^Artigo\s+(?:(?P<num>\d+)\.?\s*º?(?P<suffix>(?:-[A-Z]+)*)|(?P<unico>único))\s*$", re.IGNORECASE)


def _clean_note(s: str) -> str:
    # "Alterado pelo/a Artigo 7.º do/a [L]Lei n.º 47/2014  - Diário …" → sem marcadores de ligação nem espaços duplos
    s = re.sub(r"\[/?[A-Z]\]", "", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def articles_from_consolidated(data: dict, *, diploma_id: str, dre_id: str, diploma_title: str, url: str,
                               last_amended: str = "") -> list[Article]:
    """Converte a resposta de `DataActionGetData` em artigos, no mesmo formato do leitor do espelho."""
    try:
        frags = data["LegConsBase"]["List"]
    except (KeyError, TypeError) as e:
        raise ServiceChanged("LegConsBase em falta na resposta") from e
    stack: dict[int, str] = {}
    out: list[Article] = []
    for x in frags:
        try:
            fv, cf, nivel = x["FragmentoVersao"], x["ConsolidacaoFragmento"], int(x["Nivel"])
        except (KeyError, TypeError, ValueError) as e:
            raise ServiceChanged("fragmento com formato desconhecido") from e
        if nivel == 0:
            continue  # raiz "Diploma" (preâmbulo/sumário)
        for k in [k for k in stack if k >= nivel]:
            del stack[k]
        titulo = (fv.get("Tituo") or cf.get("Name") or "").strip()
        epigrafe = (fv.get("Epigrafe") or cf.get("Epigrafe") or "").strip()
        m = _ART_RE.match(titulo)
        if str(fv.get("TipoFragmentoId")) != TIPO_ARTIGO or not m:
            stack[nivel] = titulo + (f" — {epigrafe}" if epigrafe else "")
            continue
        path = [stack[k] for k in sorted(stack)]
        if m.group("unico"):
            number, number_int = "único", None
        else:
            number, number_int = f"{m.group('num')}.º{m.group('suffix') or ''}", int(m.group("num"))
        paragraphs = [ln.strip() for ln in (fv.get("Texto") or "").splitlines() if ln.strip()]
        notes = [_clean_note(cf["AssociacaoDescription"])] if cf.get("AssociacaoDescription") else []
        if str(fv.get("VersaoEstadoId")) == ESTADO_REVOGADO:
            # O DRE mantém o texto antigo em `Texto` nos artigos revogados: nunca o indexar como se estivesse em vigor.
            paragraphs = ["(Revogado)"]
        if not paragraphs:
            continue
        out.append(Article(
            diploma_id=diploma_id, dre_id=str(dre_id), diploma_title=diploma_title, number=number, number_int=number_int,
            epigrafe=epigrafe, hierarchy=path, scope=annex_scope(path), paragraphs=paragraphs, notes=notes,
            url=url, last_amended=last_amended, text_version="consolidado",
        ))
    return out
