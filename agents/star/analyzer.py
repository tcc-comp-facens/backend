"""
Agente Analisador da arquitetura estrela.

Cruza despesas por subfunção (SIOPS) com indicadores de saúde (DataSUS),
calcula correlações de Pearson por subfunção, identifica anomalias nos
padrões de gasto vs resultado, e gera análise textual com streaming
de chunks para ws_queue.

Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

from __future__ import annotations

import logging
import math
import os
import time
from queue import Queue
from typing import Any

from agents.base import AgenteBDI, IntentionFailure

logger = logging.getLogger(__name__)

# Mapeamento subfunção → tipo de indicador (Reqs 4.2, 4.3, 4.4)
SUBFUNCAO_INDICADOR_MAP: dict[int, list[str]] = {
    301: ["vacinacao"],          # Atenção Básica ↔ cobertura vacinal
    302: ["internacoes"],        # Assistência Hospitalar ↔ internações
    305: ["dengue", "covid"],    # Vigilância Epidemiológica ↔ dengue/COVID
}

SUBFUNCAO_NOMES: dict[int, str] = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}

CHUNK_SIZE = 80  # approximate chars per streaming chunk


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Calculate Pearson correlation coefficient.

    Returns 0.0 when there are fewer than 2 data points or zero variance.
    Result is clamped to [-1, 1].
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if den_x == 0.0 or den_y == 0.0:
        return 0.0

    r = num / (den_x * den_y)
    return max(-1.0, min(1.0, r))


def _classify(r: float) -> str:
    """Classify correlation strength as alta/média/baixa."""
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "alta"
    if abs_r >= 0.4:
        return "média"
    return "baixa"


class AgenteAnalisadorStar(AgenteBDI):
    """Agente analisador da topologia estrela.

    Recebe dados do OrquestradorEstrela (despesas + indicadores), cruza-os
    por subfunção, calcula correlações de Pearson, identifica anomalias e
    gera análise textual com streaming de chunks via ws_queue.

    WSEvent format sent to ws_queue::

        {"analysisId": str, "architecture": "star", "type": "chunk", "payload": str}
        {"analysisId": str, "architecture": "star", "type": "done",  "payload": ""}
    """

    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Return current beliefs as perception (data set by orchestrator)."""
        return {
            "despesas": self.beliefs.get("despesas", []),
            "indicadores": self.beliefs.get("indicadores", []),
            "analysis_id": self.beliefs.get("analysis_id"),
        }

    def deliberate(self) -> list[dict]:
        """Determine desires based on available data."""
        desires: list[dict] = []
        if self.beliefs.get("despesas") and self.beliefs.get("indicadores"):
            desires.append({"goal": "cruzar_dados"})
            desires.append({"goal": "calcular_correlacoes"})
            desires.append({"goal": "identificar_anomalias"})
            desires.append({"goal": "gerar_analise"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Execute a single intention."""
        goal = intention["desire"]["goal"]
        try:
            if goal == "cruzar_dados":
                self._cross_data()
            elif goal == "calcular_correlacoes":
                self._compute_correlations()
            elif goal == "identificar_anomalias":
                self._detect_anomalies()
            elif goal == "gerar_analise":
                self._stream_analysis()
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    # -- Public API called by orchestrator ------------------------------

    def analyze(
        self,
        analysis_id: str,
        data: dict[str, Any],
        ws_queue: Queue,
    ) -> dict[str, Any]:
        """Run the full analysis pipeline.

        Called by OrquestradorEstrela after the consultant returns data.

        Args:
            analysis_id: UUID of the current analysis.
            data: Dict with "despesas" and "indicadores" lists.
            ws_queue: Queue for streaming WSEvent dicts to the WebSocket server.

        Returns:
            Dict with "correlacoes", "anomalias", and "texto_analise".
        """
        self.update_beliefs({
            "analysis_id": analysis_id,
            "despesas": data.get("despesas", []),
            "indicadores": data.get("indicadores", []),
            "ws_queue": ws_queue,
        })

        self.run_cycle()

        return {
            "correlacoes": self.beliefs.get("correlacoes", []),
            "anomalias": self.beliefs.get("anomalias", []),
            "texto_analise": self.beliefs.get("texto_analise", ""),
        }

    # -- Internal pipeline steps ----------------------------------------

    def _cross_data(self) -> None:
        """Cross despesas with indicadores by subfuncao mapping (Req 4.1)."""
        despesas = self.beliefs["despesas"]
        indicadores = self.beliefs["indicadores"]

        crossed: list[dict] = []
        for subfuncao, tipos in SUBFUNCAO_INDICADOR_MAP.items():
            desp_by_year: dict[int, float] = {}
            for d in despesas:
                if d.get("subfuncao") == subfuncao:
                    desp_by_year[d["ano"]] = d["valor"]

            for tipo in tipos:
                ind_by_year: dict[int, float] = {}
                for i in indicadores:
                    if i.get("tipo") == tipo:
                        ind_by_year[i["ano"]] = i["valor"]

                common_years = sorted(set(desp_by_year) & set(ind_by_year))
                for year in common_years:
                    crossed.append({
                        "subfuncao": subfuncao,
                        "subfuncao_nome": SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                        "tipo_indicador": tipo,
                        "ano": year,
                        "valor_despesa": desp_by_year[year],
                        "valor_indicador": ind_by_year[year],
                    })

        self.beliefs["dados_cruzados"] = crossed
        logger.info("Agent %s: crossed %d data points", self.agent_id, len(crossed))

    def _compute_correlations(self) -> None:
        """Compute Pearson correlations per subfuncao-indicador pair (Reqs 4.2-4.4)."""
        crossed = self.beliefs.get("dados_cruzados", [])
        correlacoes: list[dict] = []

        pairs: dict[tuple[int, str], list[dict]] = {}
        for item in crossed:
            key = (item["subfuncao"], item["tipo_indicador"])
            pairs.setdefault(key, []).append(item)

        for (subfuncao, tipo), items in pairs.items():
            xs = [it["valor_despesa"] for it in items]
            ys = [it["valor_indicador"] for it in items]
            r = _pearson(xs, ys)
            correlacoes.append({
                "subfuncao": subfuncao,
                "subfuncao_nome": SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                "tipo_indicador": tipo,
                "correlacao": round(r, 4),
                "classificacao": _classify(r),
                "n_pontos": len(items),
            })

        self.beliefs["correlacoes"] = correlacoes
        logger.info("Agent %s: computed %d correlations", self.agent_id, len(correlacoes))

    def _detect_anomalies(self) -> None:
        """Identify anomalies in spending vs health outcomes (Req 4.6).

        An anomaly is flagged when spending is above the median for a
        subfuncao but the health indicator is below the median (or vice-versa),
        suggesting inefficiency or unexpected patterns.
        """
        crossed = self.beliefs.get("dados_cruzados", [])
        anomalias: list[dict] = []

        pairs: dict[tuple[int, str], list[dict]] = {}
        for item in crossed:
            key = (item["subfuncao"], item["tipo_indicador"])
            pairs.setdefault(key, []).append(item)

        for (subfuncao, tipo), items in pairs.items():
            if len(items) < 2:
                continue

            despesas_vals = sorted(it["valor_despesa"] for it in items)
            indicador_vals = sorted(it["valor_indicador"] for it in items)
            med_desp = despesas_vals[len(despesas_vals) // 2]
            med_ind = indicador_vals[len(indicador_vals) // 2]

            for it in items:
                high_spend = it["valor_despesa"] > med_desp
                low_outcome = it["valor_indicador"] < med_ind
                low_spend = it["valor_despesa"] < med_desp
                high_outcome = it["valor_indicador"] > med_ind

                if high_spend and low_outcome:
                    anomalias.append({
                        "subfuncao": subfuncao,
                        "subfuncao_nome": SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                        "tipo_indicador": tipo,
                        "ano": it["ano"],
                        "tipo_anomalia": "alto_gasto_baixo_resultado",
                        "descricao": (
                            f"Gasto acima da mediana em {SUBFUNCAO_NOMES.get(subfuncao, subfuncao)} "
                            f"({it['valor_despesa']:.2f}) mas indicador {tipo} abaixo da mediana "
                            f"({it['valor_indicador']:.2f}) em {it['ano']}"
                        ),
                    })
                elif low_spend and high_outcome:
                    anomalias.append({
                        "subfuncao": subfuncao,
                        "subfuncao_nome": SUBFUNCAO_NOMES.get(subfuncao, str(subfuncao)),
                        "tipo_indicador": tipo,
                        "ano": it["ano"],
                        "tipo_anomalia": "baixo_gasto_alto_resultado",
                        "descricao": (
                            f"Gasto abaixo da mediana em {SUBFUNCAO_NOMES.get(subfuncao, subfuncao)} "
                            f"({it['valor_despesa']:.2f}) mas indicador {tipo} acima da mediana "
                            f"({it['valor_indicador']:.2f}) em {it['ano']}"
                        ),
                    })

        self.beliefs["anomalias"] = anomalias
        logger.info("Agent %s: detected %d anomalies", self.agent_id, len(anomalias))

    def _stream_analysis(self) -> None:
        """Generate textual analysis and stream chunks to ws_queue (Req 4.5)."""
        ws_queue: Queue | None = self.beliefs.get("ws_queue")
        analysis_id = self.beliefs.get("analysis_id", "")
        text = self._generate_analysis_text()
        self.beliefs["texto_analise"] = text

        if ws_queue is not None:
            self._stream_text(text, analysis_id, ws_queue)

    def _generate_analysis_text(self) -> str:
        """Build analysis text from computed correlations and anomalies.

        If LLM_API_KEY is set, this could call an external LLM. The default
        implementation generates a structured analysis from the computed data
        so the system works without any external dependency.
        """
        if os.environ.get("LLM_API_KEY"):
            return self._generate_via_llm()

        return self._generate_structured_text()

    def _generate_structured_text(self) -> str:
        """Generate a structured analysis text from correlations and anomalies."""
        correlacoes = self.beliefs.get("correlacoes", [])
        anomalias = self.beliefs.get("anomalias", [])

        sections: list[str] = []
        sections.append("=== Análise de Gastos em Saúde vs Indicadores — Sorocaba-SP ===\n")

        if not correlacoes:
            sections.append(
                "Não foram encontrados dados suficientes para calcular correlações "
                "entre despesas e indicadores de saúde no período selecionado.\n"
            )
            return "\n".join(sections)

        # Correlations section
        sections.append("--- Correlações por Subfunção ---\n")
        for c in correlacoes:
            direction = "positiva" if c["correlacao"] >= 0 else "negativa"
            sections.append(
                f"• {c['subfuncao_nome']} (subfunção {c['subfuncao']}) × {c['tipo_indicador']}: "
                f"r = {c['correlacao']:.4f} — correlação {c['classificacao']} {direction} "
                f"({c['n_pontos']} pontos de dados)\n"
            )

        # Insights
        sections.append("\n--- Principais Insights ---\n")
        high_corr = [c for c in correlacoes if c["classificacao"] == "alta"]
        if high_corr:
            for c in high_corr:
                if c["correlacao"] > 0:
                    sections.append(
                        f"O aumento nos gastos com {c['subfuncao_nome']} está fortemente "
                        f"associado ao aumento de {c['tipo_indicador']}.\n"
                    )
                else:
                    sections.append(
                        f"O aumento nos gastos com {c['subfuncao_nome']} está fortemente "
                        f"associado à redução de {c['tipo_indicador']}.\n"
                    )

        low_corr = [c for c in correlacoes if c["classificacao"] == "baixa"]
        if low_corr:
            sections.append(
                "Algumas subfunções apresentam correlação baixa com seus indicadores, "
                "sugerindo que outros fatores podem influenciar os resultados de saúde.\n"
            )

        # Anomalies section
        if anomalias:
            sections.append("\n--- Anomalias Identificadas ---\n")
            for a in anomalias:
                sections.append(f"⚠ {a['descricao']}\n")

            alto_gasto = [a for a in anomalias if a["tipo_anomalia"] == "alto_gasto_baixo_resultado"]
            if alto_gasto:
                sections.append(
                    f"\nForam identificados {len(alto_gasto)} caso(s) de alto gasto com "
                    "baixo resultado, o que pode indicar ineficiência na alocação de recursos.\n"
                )
        else:
            sections.append(
                "\nNenhuma anomalia significativa foi identificada nos padrões de gasto "
                "vs resultado no período analisado.\n"
            )

        sections.append("\n=== Fim da Análise ===")
        return "\n".join(sections)

    def _generate_via_llm(self) -> str:
        """Placeholder for LLM-based analysis generation.

        When LLM_API_KEY is available, this method could call an external LLM
        API to generate richer analysis text. For now it falls back to the
        structured text generator.
        """
        logger.info("Agent %s: LLM_API_KEY detected, but using structured fallback", self.agent_id)
        return self._generate_structured_text()

    def _stream_text(self, text: str, analysis_id: str, ws_queue: Queue) -> None:
        """Stream text in chunks to ws_queue as WSEvent dicts."""
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i : i + CHUNK_SIZE]
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "chunk",
                "payload": chunk,
            })

        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
