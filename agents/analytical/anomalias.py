"""
Agente Analítico — Detecção de Anomalias.

Detecta ineficiências nos gastos públicos em saúde comparando despesas
e indicadores com suas respectivas medianas por par subfunção-indicador.

Anomalias detectadas:
- "alto_gasto_baixo_resultado": despesa acima da mediana com indicador abaixo
- "baixo_gasto_alto_resultado": despesa abaixo da mediana com indicador acima

Opera sobre dados em memória (CrossedDataPoint dicts) — sem dependência
de Neo4j ou outros serviços externos.

Requisitos: 6.1, 6.2, 6.3, 6.4, 6.5
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import AgenteBDI, IntentionFailure

logger = logging.getLogger(__name__)

SUBFUNCAO_NOMES: dict[int, str] = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}


def _median(values: list[float]) -> float:
    """Calculate the median of a list of floats.

    For even-length lists, returns the average of the two middle values.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


class AgenteAnomalias(AgenteBDI):
    """Agente analítico que detecta ineficiências via mediana (Req 6).

    Recebe dados cruzados (despesas × indicadores por subfunção e ano)
    e identifica anomalias onde o gasto e o resultado divergem da mediana
    do par subfunção-indicador.

    Tipos de anomalia:
    - "alto_gasto_baixo_resultado": despesa > mediana E indicador < mediana (Req 6.1)
    - "baixo_gasto_alto_resultado": despesa < mediana E indicador > mediana (Req 6.2)

    Pares com menos de 2 pontos de dados são ignorados (Req 6.5).
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
            desires.append({"goal": "detectar_anomalias"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Generate intentions from desires."""
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Execute a single intention."""
        goal = intention["desire"]["goal"]
        try:
            if goal == "detectar_anomalias":
                self._detect_anomalies()
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    # -- Public API called by orchestrator/supervisor -------------------

    def detect(self, dados_cruzados: list[dict]) -> list[dict]:
        """Detecta anomalias para cada par subfunção-indicador.

        Recebe dados cruzados (CrossedDataPoint dicts) e retorna lista
        de anomalias com tipo e descrição em português.

        Args:
            dados_cruzados: Lista de dicts com keys: subfuncao, subfuncao_nome,
                tipo_indicador, ano, valor_despesa, valor_indicador.

        Returns:
            Lista de dicts com keys: subfuncao, tipo_indicador, ano,
            tipo_anomalia, descricao.
            Retorna lista vazia se input for vazio.
            Ignora pares com < 2 pontos de dados (Req 6.5).
        """
        self.update_beliefs({"dados_cruzados": dados_cruzados})
        self.run_cycle()
        return self.beliefs.get("anomalias", [])

    # -- Internal computation -------------------------------------------

    def _detect_anomalies(self) -> None:
        """Detect anomalies per subfuncao-indicador pair (Reqs 6.1-6.5)."""
        crossed = self.beliefs.get("dados_cruzados", [])
        anomalias: list[dict[str, Any]] = []

        # Group data points by (subfuncao, tipo_indicador)
        pairs: dict[tuple[int, str], list[dict]] = {}
        for item in crossed:
            key = (item["subfuncao"], item["tipo_indicador"])
            pairs.setdefault(key, []).append(item)

        for (subfuncao, tipo), items in pairs.items():
            # Req 6.5: Ignore pairs with < 2 data points
            if len(items) < 2:
                continue

            despesas_vals = [it["valor_despesa"] for it in items]
            indicador_vals = [it["valor_indicador"] for it in items]

            med_desp = _median(despesas_vals)
            med_ind = _median(indicador_vals)

            subfuncao_nome = items[0].get(
                "subfuncao_nome",
                SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
            )

            for it in items:
                high_spend = it["valor_despesa"] > med_desp
                low_outcome = it["valor_indicador"] < med_ind
                low_spend = it["valor_despesa"] < med_desp
                high_outcome = it["valor_indicador"] > med_ind

                if high_spend and low_outcome:
                    # Req 6.1: alto gasto com baixo resultado
                    anomalias.append({
                        "subfuncao": subfuncao,
                        "tipo_indicador": tipo,
                        "ano": it["ano"],
                        "tipo_anomalia": "alto_gasto_baixo_resultado",
                        "descricao": (
                            f"Subfunção {subfuncao} ({subfuncao_nome}) "
                            f"em {it['ano']}: gasto acima da mediana "
                            f"(R$ {it['valor_despesa']:,.2f}) com indicador "
                            f"abaixo da mediana ({it['valor_indicador']:.1f})"
                        ),
                    })
                elif low_spend and high_outcome:
                    # Req 6.2: baixo gasto com alto resultado
                    anomalias.append({
                        "subfuncao": subfuncao,
                        "tipo_indicador": tipo,
                        "ano": it["ano"],
                        "tipo_anomalia": "baixo_gasto_alto_resultado",
                        "descricao": (
                            f"Subfunção {subfuncao} ({subfuncao_nome}) "
                            f"em {it['ano']}: gasto abaixo da mediana "
                            f"(R$ {it['valor_despesa']:,.2f}) com indicador "
                            f"acima da mediana ({it['valor_indicador']:.1f})"
                        ),
                    })

        self.beliefs["anomalias"] = anomalias
        logger.info(
            "Agent %s: detected %d anomalies", self.agent_id, len(anomalias)
        )
