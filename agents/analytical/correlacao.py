"""
Agente Analítico — Correlação Estatística.

Calcula correlações de Pearson, Spearman e Kendall Tau-b entre gastos
públicos em saúde (despesas por subfunção) e indicadores de saúde,
classificando a força da correlação com base no coeficiente de Spearman.

Opera sobre dados em memória (CrossedDataPoint dicts) — sem dependência
de Neo4j ou outros serviços externos.

Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import logging
import math
from typing import Any

from scipy import stats

from agents.base import AgenteBDI, IntentionFailure

logger = logging.getLogger(__name__)


def _safe_correlation(func, xs: list[float], ys: list[float]) -> float:
    """Run a scipy correlation function, returning 0.0 on NaN/error.

    Handles constant-value arrays (scipy returns NaN) and other edge cases.
    Result is clamped to [-1, 1].
    """
    try:
        result = func(xs, ys)
        # scipy correlation functions return (statistic, pvalue)
        r = result.statistic if hasattr(result, "statistic") else result[0]
        if math.isnan(r):
            return 0.0
        return max(-1.0, min(1.0, r))
    except Exception:
        return 0.0


def _classify(r: float) -> str:
    """Classify correlation strength as alta/média/baixa (Req 5.4).

    Uses absolute value of Spearman coefficient:
      |r| >= 0.7 → "alta"
      |r| >= 0.4 → "média"
      else       → "baixa"
    """
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "alta"
    if abs_r >= 0.4:
        return "média"
    return "baixa"


class AgenteCorrelacao(AgenteBDI):
    """Agente analítico que calcula correlações estatísticas (Req 5).

    Recebe dados cruzados (despesas × indicadores por subfunção e ano)
    e calcula três métricas de correlação para cada par subfunção-indicador:
    Pearson (linear), Spearman (rank/monotônico) e Kendall Tau-b (pares).

    Classifica a força da correlação usando Spearman como referência principal.
    Retorna 0.0 para pares com menos de 2 pontos de dados (Req 5.7).
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Return current beliefs as perception (data set by caller)."""
        return {
            "dados_cruzados": self.beliefs.get("dados_cruzados", []),
        }

    def deliberate(self) -> list[dict]:
        """Determine desires based on available data."""
        desires: list[dict] = []
        if self.beliefs.get("dados_cruzados"):
            desires.append({"goal": "calcular_correlacoes"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Generate intentions from desires."""
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Execute a single intention."""
        goal = intention["desire"]["goal"]
        try:
            if goal == "calcular_correlacoes":
                self._compute_correlations()
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    # -- Public API called by orchestrator/supervisor -------------------

    def compute(self, dados_cruzados: list[dict]) -> list[dict]:
        """Calcula correlações para cada par subfunção-indicador.

        Recebe dados cruzados (CrossedDataPoint dicts) e retorna lista
        de correlações com Pearson, Spearman, Kendall e classificação.

        Args:
            dados_cruzados: Lista de dicts com keys: subfuncao, subfuncao_nome,
                tipo_indicador, ano, valor_despesa, valor_indicador.

        Returns:
            Lista de dicts com keys: subfuncao, tipo_indicador, pearson,
            spearman, kendall, classificacao, n_pontos.
            Retorna lista vazia se input for vazio.
            Retorna 0.0 para todas as métricas se < 2 pontos (Req 5.7).
        """
        self.update_beliefs({"dados_cruzados": dados_cruzados})
        self.run_cycle()
        return self.beliefs.get("correlacoes", [])

    # -- Internal computation -------------------------------------------

    def _compute_correlations(self) -> None:
        """Compute correlations per subfuncao-indicador pair (Reqs 5.1-5.7)."""
        crossed = self.beliefs.get("dados_cruzados", [])
        correlacoes: list[dict[str, Any]] = []

        # Group data points by (subfuncao, tipo_indicador)
        pairs: dict[tuple[int, str], list[dict]] = {}
        for item in crossed:
            key = (item["subfuncao"], item["tipo_indicador"])
            pairs.setdefault(key, []).append(item)

        for (subfuncao, tipo), items in pairs.items():
            n = len(items)
            xs = [it["valor_despesa"] for it in items]
            ys = [it["valor_indicador"] for it in items]

            if n < 2:
                # Req 5.7: Return 0.0 for all metrics when < 2 data points
                correlacoes.append({
                    "subfuncao": subfuncao,
                    "tipo_indicador": tipo,
                    "pearson": 0.0,
                    "spearman": 0.0,
                    "kendall": 0.0,
                    "classificacao": "baixa",
                    "n_pontos": n,
                })
            else:
                # Req 5.1: Pearson (linear correlation)
                r_pearson = _safe_correlation(stats.pearsonr, xs, ys)
                # Req 5.2: Spearman (rank correlation)
                r_spearman = _safe_correlation(stats.spearmanr, xs, ys)
                # Req 5.3: Kendall Tau-b (pair concordance)
                r_kendall = _safe_correlation(stats.kendalltau, xs, ys)

                r_spearman_rounded = round(r_spearman, 4)
                correlacoes.append({
                    "subfuncao": subfuncao,
                    "tipo_indicador": tipo,
                    "pearson": round(r_pearson, 4),
                    "spearman": r_spearman_rounded,
                    "kendall": round(r_kendall, 4),
                    "classificacao": _classify(r_spearman_rounded),  # Req 5.4
                    "n_pontos": n,
                })

        self.beliefs["correlacoes"] = correlacoes
        logger.info(
            "Agent %s: computed %d correlations", self.agent_id, len(correlacoes)
        )
