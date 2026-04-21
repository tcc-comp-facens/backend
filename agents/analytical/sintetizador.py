"""
Agente Analítico — Sintetizador de Texto.

Gera texto consolidado de análise via LLM (Groq primário, Gemini fallback)
a partir de correlações, anomalias e contexto orçamentário. Faz streaming
do texto em chunks de ~80 caracteres via ws_queue.

Quando o LLM está indisponível após 3 tentativas, gera texto estruturado
como fallback.

Requisitos: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import logging
from queue import Queue
from typing import Any

from agents.base import AgenteBDI, IntentionFailure

logger = logging.getLogger(__name__)

CHUNK_SIZE = 80  # approximate chars per streaming chunk

SUBFUNCAO_NOMES: dict[int, str] = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}


class AgenteSintetizador(AgenteBDI):
    """Agente analítico que gera texto consolidado via LLM com streaming (Req 7).

    Recebe correlações, anomalias e contexto orçamentário dos demais agentes
    analíticos e de contexto, consolida em um prompt para o LLM, e faz
    streaming do texto gerado em chunks via ws_queue.

    WSEvent format sent to ws_queue::

        {"analysisId": str, "architecture": str, "type": "chunk", "payload": str}
        {"analysisId": str, "architecture": str, "type": "done",  "payload": ""}
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Return current beliefs as perception (data set by caller)."""
        return {
            "correlacoes": self.beliefs.get("correlacoes", []),
            "anomalias": self.beliefs.get("anomalias", []),
            "contexto_orcamentario": self.beliefs.get("contexto_orcamentario", {}),
            "analysis_id": self.beliefs.get("analysis_id"),
            "architecture": self.beliefs.get("architecture"),
        }

    def deliberate(self) -> list[dict]:
        """Determine desires based on available data."""
        desires: list[dict] = []
        # We can synthesize even with empty correlacoes/anomalias (fallback text)
        if self.beliefs.get("analysis_id") is not None:
            desires.append({"goal": "sintetizar_texto"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Generate intentions from desires."""
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Execute a single intention."""
        goal = intention["desire"]["goal"]
        try:
            if goal == "sintetizar_texto":
                self._synthesize_and_stream()
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    # -- Public API called by orchestrator/supervisor -------------------

    def synthesize(
        self,
        correlacoes: list[dict],
        anomalias: list[dict],
        contexto_orcamentario: dict[str, Any],
        analysis_id: str,
        ws_queue: Queue,
        architecture: str,
        data_coverage: dict[str, Any] | None = None,
    ) -> str:
        """Gera e faz streaming do texto de análise.

        Usa LLM (Groq/Gemini) com fallback para texto estruturado (Req 7.3).
        Streaming em chunks de ~80 chars (Req 7.2).
        Envia evento "done" ao concluir (Req 7.6).

        Args:
            correlacoes: Lista de dicts com correlações calculadas.
            anomalias: Lista de dicts com anomalias detectadas.
            contexto_orcamentario: Dict com tendências orçamentárias por subfunção.
            analysis_id: UUID da análise corrente.
            ws_queue: Queue para streaming de WSEvent dicts.
            architecture: "star" ou "hierarchical".
            data_coverage: Dict com cobertura de dados e gaps detectados.

        Returns:
            Texto completo da análise gerada.
        """
        self.update_beliefs({
            "correlacoes": correlacoes,
            "anomalias": anomalias,
            "contexto_orcamentario": contexto_orcamentario,
            "analysis_id": analysis_id,
            "ws_queue": ws_queue,
            "architecture": architecture,
            "data_coverage": data_coverage or {},
        })

        self.run_cycle()

        return self.beliefs.get("texto_analise", "")

    # -- Internal pipeline steps ----------------------------------------

    def _synthesize_and_stream(self) -> None:
        """Generate textual analysis and stream chunks to ws_queue (Req 7.1, 7.2)."""
        ws_queue: Queue | None = self.beliefs.get("ws_queue")
        analysis_id = self.beliefs.get("analysis_id", "")
        architecture = self.beliefs.get("architecture", "")

        text = self._generate_analysis_text()
        self.beliefs["texto_analise"] = text

        if ws_queue is not None:
            self._stream_text(text, analysis_id, ws_queue, architecture)

    def _generate_analysis_text(self) -> str:
        """Build analysis text via LLM or structured fallback (Req 7.1, 7.3).

        Tries LLM (Groq primary, Gemini fallback via llm_client.generate())
        first. If LLM returns None (unavailable after 3 retries), falls back
        to structured text.
        """
        try:
            return self._generate_via_llm()
        except Exception:
            logger.warning(
                "Agent %s: LLM generation failed, using structured fallback",
                self.agent_id,
            )
            return self._generate_structured_text()

    def _generate_via_llm(self) -> str:
        """Generate analysis text using LLM via centralized client (Req 7.1)."""
        import llm_client

        correlacoes = self.beliefs.get("correlacoes", [])
        anomalias = self.beliefs.get("anomalias", [])
        contexto_orcamentario = self.beliefs.get("contexto_orcamentario", {})

        prompt = self._build_prompt(correlacoes, anomalias, contexto_orcamentario)

        text = llm_client.generate(prompt)
        if text:
            logger.info("Agent %s: LLM analysis generated successfully", self.agent_id)
            return text

        logger.warning("Agent %s: LLM unavailable, using fallback", self.agent_id)
        return self._generate_structured_text()

    def _build_prompt(
        self,
        correlacoes: list[dict],
        anomalias: list[dict],
        contexto_orcamentario: dict[str, Any],
    ) -> str:
        """Build the LLM prompt from analysis data (Req 7.4)."""
        data_coverage = self.beliefs.get("data_coverage", {})
        gaps = data_coverage.get("gaps", [])
        summary = data_coverage.get("summary", {})

        coverage_section = ""
        if gaps:
            coverage_section = (
                "\nAVISO — DADOS FALTANTES:\n"
                f"Completude das despesas: {summary.get('despesas_completeness', 1.0):.0%}\n"
                f"Completude dos indicadores: {summary.get('indicadores_completeness', 1.0):.0%}\n"
                "Lacunas detectadas:\n"
            )
            for g in gaps:
                coverage_section += f"- {g['description']}\n"
            coverage_section += (
                "\nIMPORTANTE: Mencione explicitamente na análise quais dados "
                "estão faltando e como isso limita as conclusões.\n"
            )

        return (
            "Você é um analista de políticas públicas de saúde. "
            "Analise os dados abaixo sobre gastos em saúde vs indicadores "
            "de saúde do município de Sorocaba-SP e gere uma análise detalhada "
            "em português brasileiro.\n\n"
            "Foram calculadas três métricas de correlação para cada par:\n"
            "- Pearson: mede relação linear\n"
            "- Spearman: mede relação monotônica (baseada em ranks, robusta a outliers)\n"
            "- Kendall Tau: mede concordância entre pares (ideal para amostras pequenas)\n\n"
            f"Correlações encontradas:\n{correlacoes}\n\n"
            f"Anomalias identificadas:\n{anomalias}\n\n"
            f"Contexto orçamentário (tendências de gasto):\n{contexto_orcamentario}\n\n"
            f"{coverage_section}\n"
            "Gere uma análise completa com as seguintes seções:\n"
            "1. Resumo Executivo\n"
            "2. Cobertura de Dados (quais dados estão disponíveis e quais estão faltando)\n"
            "3. Análise das Correlações (comparando os três métodos e explicando divergências)\n"
            "4. Discussão das Anomalias\n"
            "5. Contexto Orçamentário\n"
            "Use formatação com seções claras."
        )

    def _generate_structured_text(self) -> str:
        """Generate structured fallback text when LLM is unavailable (Req 7.3, 7.4).

        Includes: resumo executivo, correlações analysis, anomalias discussion,
        and contexto orçamentário.
        """
        correlacoes = self.beliefs.get("correlacoes", [])
        anomalias = self.beliefs.get("anomalias", [])
        contexto_orcamentario = self.beliefs.get("contexto_orcamentario", {})

        sections: list[str] = []

        # --- Resumo Executivo (Req 7.4) ---
        sections.append("=== Resumo Executivo ===\n")
        sections.append(
            "Análise consolidada dos gastos em saúde vs indicadores de saúde "
            "do município de Sorocaba-SP, integrando correlações estatísticas, "
            "detecção de anomalias e contexto orçamentário.\n"
        )
        sections.append(
            f"Foram analisadas {len(correlacoes)} correlação(ões) e "
            f"identificadas {len(anomalias)} anomalia(s).\n"
        )

        # --- Cobertura de Dados ---
        data_coverage = self.beliefs.get("data_coverage", {})
        gaps = data_coverage.get("gaps", [])
        summary = data_coverage.get("summary", {})

        sections.append("\n=== Cobertura de Dados ===\n")
        if not gaps:
            sections.append(
                "Todos os dados esperados para o período estão disponíveis. "
                "Não foram detectadas lacunas.\n"
            )
        else:
            desp_pct = summary.get("despesas_completeness", 1.0)
            ind_pct = summary.get("indicadores_completeness", 1.0)
            sections.append(
                f"Completude das despesas SIOPS: {desp_pct:.0%}\n"
                f"Completude dos indicadores DataSUS: {ind_pct:.0%}\n"
                f"Total de lacunas detectadas: {len(gaps)}\n\n"
            )
            sections.append("Lacunas identificadas:\n")
            for g in gaps:
                sections.append(f"⚠ {g['description']}\n")
            sections.append(
                "\nNota: As correlações e anomalias foram calculadas apenas "
                "com os dados disponíveis. Anos sem dados em ambos os lados "
                "(despesa e indicador) foram excluídos do cálculo, o que pode "
                "reduzir a significância estatística dos resultados.\n"
            )

        # --- Análise das Correlações (Req 7.4) ---
        sections.append("\n=== Análise das Correlações ===\n")
        if not correlacoes:
            sections.append(
                "Não foram encontrados dados suficientes para calcular correlações "
                "entre despesas e indicadores de saúde no período selecionado.\n"
            )
        else:
            for c in correlacoes:
                subfuncao_nome = c.get(
                    "subfuncao_nome",
                    SUBFUNCAO_NOMES.get(c.get("subfuncao", 0), str(c.get("subfuncao", ""))),
                )
                direction = "positiva" if c.get("spearman", 0) >= 0 else "negativa"
                classificacao = c.get("classificacao", "baixa")
                sections.append(
                    f"• {subfuncao_nome} × {c.get('tipo_indicador', '')}:\n"
                    f"  Pearson: {c.get('pearson', 0):.4f} | "
                    f"Spearman: {c.get('spearman', 0):.4f} | "
                    f"Kendall: {c.get('kendall', 0):.4f}\n"
                    f"  Classificação: {classificacao} {direction} "
                    f"({c.get('n_pontos', 0)} pontos de dados)\n"
                )

            # Insights
            high_corr = [c for c in correlacoes if c.get("classificacao") == "alta"]
            if high_corr:
                sections.append("\nCorrelações fortes identificadas:\n")
                for c in high_corr:
                    subfuncao_nome = c.get(
                        "subfuncao_nome",
                        SUBFUNCAO_NOMES.get(c.get("subfuncao", 0), str(c.get("subfuncao", ""))),
                    )
                    if c.get("spearman", 0) > 0:
                        sections.append(
                            f"  O aumento nos gastos com {subfuncao_nome} está fortemente "
                            f"associado ao aumento de {c.get('tipo_indicador', '')}.\n"
                        )
                    else:
                        sections.append(
                            f"  O aumento nos gastos com {subfuncao_nome} está fortemente "
                            f"associado à redução de {c.get('tipo_indicador', '')}.\n"
                        )

        # --- Discussão das Anomalias (Req 7.4) ---
        sections.append("\n=== Discussão das Anomalias ===\n")
        if not anomalias:
            sections.append(
                "Nenhuma anomalia significativa foi identificada nos padrões de gasto "
                "vs resultado no período analisado.\n"
            )
        else:
            for a in anomalias:
                sections.append(f"⚠ {a.get('descricao', 'Anomalia detectada')}\n")

            alto_gasto = [
                a for a in anomalias
                if a.get("tipo_anomalia") == "alto_gasto_baixo_resultado"
            ]
            baixo_gasto = [
                a for a in anomalias
                if a.get("tipo_anomalia") == "baixo_gasto_alto_resultado"
            ]
            if alto_gasto:
                sections.append(
                    f"\nForam identificados {len(alto_gasto)} caso(s) de alto gasto com "
                    "baixo resultado, o que pode indicar ineficiência na alocação de recursos.\n"
                )
            if baixo_gasto:
                sections.append(
                    f"\nForam identificados {len(baixo_gasto)} caso(s) de baixo gasto com "
                    "alto resultado, sugerindo eficiência ou fatores externos positivos.\n"
                )

        # --- Contexto Orçamentário (Req 7.4) ---
        sections.append("\n=== Contexto Orçamentário ===\n")
        if not contexto_orcamentario:
            sections.append(
                "Dados de contexto orçamentário não disponíveis para o período analisado.\n"
            )
        else:
            for subfuncao_key, tendencia_data in contexto_orcamentario.items():
                if isinstance(tendencia_data, dict):
                    tendencia = tendencia_data.get("tendencia", "desconhecida")
                    variacao = tendencia_data.get("variacao_media_percentual", 0)
                    anos = tendencia_data.get("anos_analisados", [])
                    subfuncao_nome = SUBFUNCAO_NOMES.get(
                        int(subfuncao_key) if str(subfuncao_key).isdigit() else 0,
                        str(subfuncao_key),
                    )
                    sections.append(
                        f"• {subfuncao_nome} (subfunção {subfuncao_key}):\n"
                        f"  Tendência: {tendencia} | "
                        f"Variação média: {variacao:.1f}% | "
                        f"Anos: {anos}\n"
                    )

        sections.append("\n=== Fim da Análise ===")
        return "\n".join(sections)

    def _stream_text(
        self,
        text: str,
        analysis_id: str,
        ws_queue: Queue,
        architecture: str,
    ) -> None:
        """Stream text in chunks to ws_queue as WSEvent dicts (Req 7.2, 7.6)."""
        for i in range(0, len(text), CHUNK_SIZE):
            chunk = text[i: i + CHUNK_SIZE]
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": architecture,
                "type": "chunk",
                "payload": chunk,
            })
