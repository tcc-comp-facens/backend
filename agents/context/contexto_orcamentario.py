"""
Agente de Contexto — Contexto Orçamentário.

Analisa tendências temporais de gasto público em saúde por subfunção,
calculando variação percentual ano a ano e classificando tendências
como crescimento, corte, estagnação ou dados insuficientes.

Opera sobre dados em memória (DespesaRecord dicts) — sem dependência
de Neo4j ou outros serviços externos.

Requisitos: 8.1, 8.2, 8.3, 8.4, 8.5
"""

from __future__ import annotations

import logging
import math
from typing import Any

from agents.base import AgenteBDI, IntentionFailure

logger = logging.getLogger(__name__)

# Threshold for stagnation classification (Req 8.2)
STAGNATION_THRESHOLD = 5.0  # |variation| < 5% → estagnação

# Minimum consecutive years for growth/cut classification (Req 8.2)
MIN_CONSECUTIVE_YEARS = 2


def _compute_yoy_variation(valor_current: float, valor_previous: float) -> float:
    """Compute year-over-year percentage variation (Req 8.1).

    Formula: ((valor_n - valor_n-1) / valor_n-1) × 100

    Handles division by zero when valor_previous == 0:
    - If valor_current > 0: returns +inf (infinite growth)
    - If valor_current < 0: returns -inf (infinite cut)
    - If valor_current == 0: returns 0.0 (no change)
    """
    if valor_previous == 0.0:
        if valor_current > 0:
            return math.inf
        elif valor_current < 0:
            return -math.inf
        else:
            return 0.0
    return ((valor_current - valor_previous) / valor_previous) * 100.0


def _classify_trend(variations: list[float]) -> str:
    """Classify trend based on consecutive variations (Req 8.2).

    Rules:
    - "crescimento": consecutive positive variations for 2+ years
    - "corte": consecutive negative variations for 2+ years
    - "estagnacao": all |variation| < 5% in absolute value

    When none of the above apply cleanly, use the dominant pattern:
    - Check for longest consecutive positive/negative streak
    - If all variations are within stagnation threshold → "estagnacao"
    - Otherwise classify by the dominant consecutive streak
    """
    if not variations:
        return "insuficiente"

    # Check if all variations are within stagnation threshold
    all_stagnant = all(
        abs(v) < STAGNATION_THRESHOLD for v in variations if math.isfinite(v)
    )
    if all_stagnant and all(math.isfinite(v) for v in variations):
        return "estagnacao"

    # Count longest consecutive positive streak
    max_positive_streak = 0
    current_positive = 0
    for v in variations:
        if v > 0 or (math.isinf(v) and v > 0):
            current_positive += 1
            max_positive_streak = max(max_positive_streak, current_positive)
        else:
            current_positive = 0

    # Count longest consecutive negative streak
    max_negative_streak = 0
    current_negative = 0
    for v in variations:
        if v < 0 or (math.isinf(v) and v < 0):
            current_negative += 1
            max_negative_streak = max(max_negative_streak, current_negative)
        else:
            current_negative = 0

    if max_positive_streak >= MIN_CONSECUTIVE_YEARS:
        return "crescimento"
    if max_negative_streak >= MIN_CONSECUTIVE_YEARS:
        return "corte"

    # Default: check average direction
    finite_vars = [v for v in variations if math.isfinite(v)]
    if finite_vars:
        avg = sum(finite_vars) / len(finite_vars)
        if abs(avg) < STAGNATION_THRESHOLD:
            return "estagnacao"
        return "crescimento" if avg > 0 else "corte"

    # All infinite — classify by sign of first infinite value
    for v in variations:
        if math.isinf(v):
            return "crescimento" if v > 0 else "corte"

    return "estagnacao"


class AgenteContextoOrcamentario(AgenteBDI):
    """Agente de contexto que analisa tendências temporais de gasto (Req 8).

    Recebe dados de despesas agregados dos agentes de domínio e calcula
    variação percentual ano a ano por subfunção, classificando tendências
    como crescimento, corte, estagnação ou dados insuficientes.

    Classificação de tendências (Req 8.2):
    - "crescimento": variação positiva consecutiva por 2+ anos
    - "corte": variação negativa consecutiva por 2+ anos
    - "estagnacao": |variação| < 5% em valor absoluto
    - "insuficiente": menos de 2 anos de dados (Req 8.5)
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Return current beliefs as perception (data set by caller)."""
        return {
            "despesas": self.beliefs.get("despesas", []),
        }

    def deliberate(self) -> list[dict]:
        """Determine desires based on available data."""
        desires: list[dict] = []
        if self.beliefs.get("despesas"):
            desires.append({"goal": "analisar_tendencias"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Generate intentions from desires."""
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Execute a single intention."""
        goal = intention["desire"]["goal"]
        try:
            if goal == "analisar_tendencias":
                self._analyze_trends()
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    # -- Public API called by orchestrator/supervisor -------------------

    def analyze_trends(self, despesas: list[dict]) -> dict[int, dict]:
        """Analisa tendências temporais de gasto por subfunção.

        Recebe despesas agregadas (DespesaRecord dicts) e retorna
        tendências por subfunção com variação média e classificação.

        Args:
            despesas: Lista de dicts com keys: subfuncao, subfuncaoNome,
                ano, valor.

        Returns:
            Dicionário com chave = subfunção (int) e valor = dict com:
            subfuncao, tendencia, variacao_media_percentual, anos_analisados.
            Retorna dict vazio se input for vazio.
            Retorna "insuficiente" para subfunções com < 2 anos (Req 8.5).
        """
        self.update_beliefs({"despesas": despesas})
        self.run_cycle()
        return self.beliefs.get("tendencias", {})

    # -- Internal computation -------------------------------------------

    def _analyze_trends(self) -> None:
        """Compute trends per subfunção (Reqs 8.1-8.5)."""
        despesas = self.beliefs.get("despesas", [])
        tendencias: dict[int, dict[str, Any]] = {}

        # Group despesas by subfunção
        by_subfuncao: dict[int, list[dict]] = {}
        for d in despesas:
            sf = d["subfuncao"]
            by_subfuncao.setdefault(sf, []).append(d)

        for subfuncao, items in by_subfuncao.items():
            # Aggregate values by year (sum if multiple entries per year)
            year_values: dict[int, float] = {}
            for item in items:
                ano = item["ano"]
                year_values[ano] = year_values.get(ano, 0.0) + item["valor"]

            # Sort by year
            sorted_years = sorted(year_values.keys())
            anos_analisados = sorted_years

            # Req 8.5: Insufficient data
            if len(sorted_years) < 2:
                tendencias[subfuncao] = {
                    "subfuncao": subfuncao,
                    "tendencia": "insuficiente",
                    "variacao_media_percentual": 0.0,
                    "anos_analisados": anos_analisados,
                }
                continue

            # Req 8.1: Calculate year-over-year variations
            variations: list[float] = []
            for i in range(1, len(sorted_years)):
                prev_year = sorted_years[i - 1]
                curr_year = sorted_years[i]
                variation = _compute_yoy_variation(
                    year_values[curr_year], year_values[prev_year]
                )
                variations.append(variation)

            # Req 8.2: Classify trend
            tendencia = _classify_trend(variations)

            # Compute average variation (use only finite values)
            finite_vars = [v for v in variations if math.isfinite(v)]
            if finite_vars:
                variacao_media = sum(finite_vars) / len(finite_vars)
            else:
                # All variations are infinite — use sign to indicate direction
                positive_inf = sum(1 for v in variations if v == math.inf)
                negative_inf = sum(1 for v in variations if v == -math.inf)
                if positive_inf > negative_inf:
                    variacao_media = math.inf
                elif negative_inf > positive_inf:
                    variacao_media = -math.inf
                else:
                    variacao_media = 0.0

            tendencias[subfuncao] = {
                "subfuncao": subfuncao,
                "tendencia": tendencia,
                "variacao_media_percentual": round(variacao_media, 2) if math.isfinite(variacao_media) else variacao_media,
                "anos_analisados": anos_analisados,
            }

        self.beliefs["tendencias"] = tendencias
        logger.info(
            "Agent %s: analyzed trends for %d subfunções",
            self.agent_id,
            len(tendencias),
        )
