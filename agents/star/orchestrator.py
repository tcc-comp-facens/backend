"""
Orquestrador central da arquitetura estrela.

Hub central que instancia e coordena 8 agentes especializados:
4 de domínio, 3 analíticos e 1 de contexto. Intermedia toda
comunicação entre agentes — nenhum agente periférico chama outro
diretamente.

Pipeline:
  1. Fase de Domínio: 4 agentes consultam Neo4j em sequência
  2. Cruzamento de dados: despesas × indicadores por subfunção
  3. Fase Analítica: contexto orçamentário, correlação, anomalias, síntese

Requisitos: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 11.1, 11.2
"""

from __future__ import annotations

import logging
import uuid
from queue import Queue
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI
from agents.data_crossing import cross_domain_data, detect_data_gaps
from agents.domain.vigilancia_epidemiologica import AgenteVigilanciaEpidemiologica
from agents.domain.saude_hospitalar import AgenteSaudeHospitalar
from agents.domain.atencao_primaria import AgenteAtencaoPrimaria
from agents.domain.mortalidade import AgenteMortalidade
from agents.analytical.correlacao import AgenteCorrelacao
from agents.analytical.anomalias import AgenteAnomalias
from agents.analytical.sintetizador import AgenteSintetizador
from agents.context.contexto_orcamentario import AgenteContextoOrcamentario
from message_counter import MessageCounter
from metrics import MetricsCollector

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class OrquestradorEstrela(AgenteBDI):
    """Hub central da topologia estrela coordenando 8 agentes especializados (Req 9).

    Toda comunicação entre agentes periféricos passa por este
    orquestrador (Req 9.3). Distribui tarefas via chamadas de
    método Python, registra métricas de tempo de execução por agente
    (Req 9.7), e trata falhas com resultados parciais (Req 9.8).

    Attributes:
        neo4j_client: Cliente Neo4j para queries e persistência.
    """

    def __init__(self, agent_id: str, neo4j_client: Neo4jClient) -> None:
        super().__init__(agent_id)
        self.neo4j_client = neo4j_client

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Percebe parâmetros da análise a partir das crenças."""
        return {
            "analysis_id": self.beliefs.get("analysis_id"),
            "date_from": self.beliefs.get("date_from"),
            "date_to": self.beliefs.get("date_to"),
            "health_params": self.beliefs.get("health_params", []),
        }

    def deliberate(self) -> list[dict]:
        """Define desejos: consultar dados, analisar e persistir métricas."""
        desires: list[dict] = []
        if self.beliefs.get("analysis_id"):
            desires.append({"goal": "executar_pipeline_dominio"})
            desires.append({"goal": "executar_pipeline_analitico"})
            desires.append({"goal": "persistir_metricas"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        return [{"desire": d, "status": "pending"} for d in desires]

    # -- Public API -----------------------------------------------------

    def run(
        self,
        analysis_id: str,
        params: dict[str, Any],
        ws_queue: Queue,
    ) -> dict[str, Any]:
        """Executa o pipeline completo da arquitetura estrela com 8 agentes.

        1. Instancia 8 agentes com IDs únicos (Req 9.1)
        2. Executa 4 agentes de domínio em sequência (Req 9.2)
        3. Agrega dados e cruza despesas × indicadores (Req 9.4)
        4. Passa despesas agregadas ao AgenteContextoOrcamentario (Req 9.4)
        5. Passa dados cruzados ao AgenteCorrelacao (Req 9.4)
        6. Passa dados cruzados ao AgenteAnomalias (Req 9.5)
        7. Passa correlações, anomalias e contexto ao AgenteSintetizador (Req 9.6)
        8. Registra métricas por agente (Req 9.7)
        9. Trata falhas com resultados parciais (Req 9.8)

        Args:
            analysis_id: UUID da análise.
            params: Dicionário com date_from, date_to, health_params.
            ws_queue: Fila para streaming de eventos WebSocket.

        Returns:
            Dicionário com resultado completo da análise.
        """
        # Configure beliefs for BDI cycle
        self.update_beliefs({
            "analysis_id": analysis_id,
            "date_from": params.get("date_from"),
            "date_to": params.get("date_to"),
            "health_params": params.get("health_params", []),
            "ws_queue": ws_queue,
        })

        date_from = params.get("date_from")
        date_to = params.get("date_to")

        # -- 1. Instanciar 8 agentes com IDs únicos (Req 9.1) --
        vig_id = f"star-vigilancia-{uuid.uuid4().hex[:8]}"
        hosp_id = f"star-hospitalar-{uuid.uuid4().hex[:8]}"
        prim_id = f"star-primaria-{uuid.uuid4().hex[:8]}"
        mort_id = f"star-mortalidade-{uuid.uuid4().hex[:8]}"
        ctx_id = f"star-contexto-{uuid.uuid4().hex[:8]}"
        corr_id = f"star-correlacao-{uuid.uuid4().hex[:8]}"
        anom_id = f"star-anomalias-{uuid.uuid4().hex[:8]}"
        sint_id = f"star-sintetizador-{uuid.uuid4().hex[:8]}"

        agente_vigilancia = AgenteVigilanciaEpidemiologica(vig_id, self.neo4j_client)
        agente_hospitalar = AgenteSaudeHospitalar(hosp_id, self.neo4j_client)
        agente_primaria = AgenteAtencaoPrimaria(prim_id, self.neo4j_client)
        agente_mortalidade = AgenteMortalidade(mort_id, self.neo4j_client)
        agente_contexto = AgenteContextoOrcamentario(ctx_id)
        agente_correlacao = AgenteCorrelacao(corr_id)
        agente_anomalias = AgenteAnomalias(anom_id)
        agente_sintetizador = AgenteSintetizador(sint_id)

        # -- 2. MessageCounter para esta análise (Req 11.1, 11.2) --
        counter = MessageCounter()

        # Collectors for metrics persistence (Req 9.7)
        collectors: list[tuple[str, str, MetricsCollector]] = []

        # Aggregated data from domain agents
        all_despesas: list[dict] = []
        all_indicadores: list[dict] = []

        # ============================================================
        # FASE 1 — Pipeline de Domínio (Req 9.2)
        # Execute 4 domain agents in SEQUENCE, collecting data from each
        # ============================================================

        domain_agents: list[tuple[str, str, Any]] = [
            (vig_id, "vigilancia_epidemiologica", agente_vigilancia),
            (hosp_id, "saude_hospitalar", agente_hospitalar),
            (prim_id, "atencao_primaria", agente_primaria),
            (mort_id, "mortalidade", agente_mortalidade),
        ]

        for agent_id_str, agent_type, agent in domain_agents:
            mc = MetricsCollector(agent_id_str, agent_type)
            mc.start()
            try:
                # Req 9.3: orchestrator intermediates ALL communication
                result = agent.query(analysis_id, date_from, date_to)
                mc.stop()
                # Req 11.1: 2 messages per call (ida + volta)
                counter.increment(2)
                all_despesas.extend(result.get("despesas", []))
                all_indicadores.extend(result.get("indicadores", []))
                logger.info(
                    "OrquestradorEstrela: %s returned %d despesas, %d indicadores",
                    agent_type,
                    len(result.get("despesas", [])),
                    len(result.get("indicadores", [])),
                )
            except Exception as exc:
                mc.stop()
                # Req 9.8: send error event, continue with partial results
                logger.error(
                    "OrquestradorEstrela: %s failed — %s", agent_type, exc
                )
                ws_queue.put({
                    "analysisId": analysis_id,
                    "architecture": "star",
                    "type": "error",
                    "payload": f"Agente {agent_type} falhou: {exc}",
                })
            collectors.append((agent_id_str, agent_type, mc))

        # ============================================================
        # Cruzamento de dados (Req 9.4)
        # ============================================================

        # Deduplicate despesas (mortalidade returns all subfunções,
        # which may overlap with other domain agents)
        seen_despesas: set[tuple[int, int]] = set()
        unique_despesas: list[dict] = []
        for d in all_despesas:
            key = (d.get("subfuncao", 0), d.get("ano", 0))
            if key not in seen_despesas:
                seen_despesas.add(key)
                unique_despesas.append(d)

        dados_cruzados = cross_domain_data(unique_despesas, all_indicadores)

        # Detect data gaps for transparency
        data_coverage = detect_data_gaps(
            unique_despesas, all_indicadores, date_from, date_to
        )
        if data_coverage["summary"]["has_gaps"]:
            logger.warning(
                "OrquestradorEstrela: %d data gaps detected",
                data_coverage["summary"]["total_gaps"],
            )

        # ============================================================
        # FASE 2 — Pipeline Analítico
        # ============================================================

        # -- Contexto Orçamentário (Req 9.4) --
        contexto_orcamentario: dict[int, dict] = {}
        mc_ctx = MetricsCollector(ctx_id, "contexto_orcamentario")
        mc_ctx.start()
        try:
            contexto_orcamentario = agente_contexto.analyze_trends(unique_despesas)
            mc_ctx.stop()
            counter.increment(2)
            logger.info(
                "OrquestradorEstrela: contexto_orcamentario computed for %d subfunções",
                len(contexto_orcamentario),
            )
        except Exception as exc:
            mc_ctx.stop()
            logger.error(
                "OrquestradorEstrela: contexto_orcamentario failed — %s", exc
            )
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "error",
                "payload": f"Agente contexto_orcamentario falhou: {exc}",
            })
        collectors.append((ctx_id, "contexto_orcamentario", mc_ctx))

        # -- Correlação (Req 9.4) --
        correlacoes: list[dict] = []
        mc_corr = MetricsCollector(corr_id, "correlacao")
        mc_corr.start()
        try:
            correlacoes = agente_correlacao.compute(dados_cruzados)
            mc_corr.stop()
            counter.increment(2)
            logger.info(
                "OrquestradorEstrela: computed %d correlações", len(correlacoes)
            )
        except Exception as exc:
            mc_corr.stop()
            logger.error(
                "OrquestradorEstrela: correlacao failed — %s", exc
            )
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "error",
                "payload": f"Agente correlacao falhou: {exc}",
            })
        collectors.append((corr_id, "correlacao", mc_corr))

        # -- Anomalias (Req 9.5) --
        anomalias: list[dict] = []
        mc_anom = MetricsCollector(anom_id, "anomalias")
        mc_anom.start()
        try:
            anomalias = agente_anomalias.detect(dados_cruzados)
            mc_anom.stop()
            counter.increment(2)
            logger.info(
                "OrquestradorEstrela: detected %d anomalias", len(anomalias)
            )
        except Exception as exc:
            mc_anom.stop()
            logger.error(
                "OrquestradorEstrela: anomalias failed — %s", exc
            )
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "error",
                "payload": f"Agente anomalias falhou: {exc}",
            })
        collectors.append((anom_id, "anomalias", mc_anom))

        # -- Sintetizador (Req 9.6) --
        texto_analise: str = ""
        mc_sint = MetricsCollector(sint_id, "sintetizador")
        mc_sint.start()
        try:
            texto_analise = agente_sintetizador.synthesize(
                correlacoes=correlacoes,
                anomalias=anomalias,
                contexto_orcamentario=contexto_orcamentario,
                analysis_id=analysis_id,
                ws_queue=ws_queue,
                architecture="star",
                data_coverage=data_coverage,
            )
            mc_sint.stop()
            counter.increment(2)
            logger.info(
                "OrquestradorEstrela: synthesis complete (%d chars)",
                len(texto_analise),
            )
        except Exception as exc:
            mc_sint.stop()
            logger.error(
                "OrquestradorEstrela: sintetizador failed — %s", exc
            )
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "error",
                "payload": f"Agente sintetizador falhou: {exc}",
            })
        collectors.append((sint_id, "sintetizador", mc_sint))

        # ============================================================
        # Persistir métricas de cada agente (Req 9.7)
        # ============================================================

        for agent_id_str, agent_type, mc in collectors:
            try:
                mc.persist(self.neo4j_client, analysis_id, "star")
            except Exception as exc:
                logger.error(
                    "OrquestradorEstrela: failed to persist metrics for %s — %s",
                    agent_id_str,
                    exc,
                )

        # Send benchmark metrics event (Req 11.2)
        agent_metrics = []
        total_time_ms = 0.0
        for _, agent_type, mc in collectors:
            try:
                m = mc.collect()
                agent_metrics.append({
                    "agentName": agent_type,
                    "executionTimeMs": m["executionTimeMs"],
                    "cpuPercent": m["cpuPercent"],
                    "memoryMb": m["memoryMb"],
                })
                total_time_ms += m["executionTimeMs"]
            except Exception:
                pass
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "metric",
            "payload": {
                "architecture": "star",
                "totalExecutionTimeMs": round(total_time_ms, 2),
                "agentMetrics": agent_metrics,
                "messageCount": counter.count,
            },
        })

        # ============================================================
        # Build and return result dict
        # ============================================================

        result = {
            "despesas": unique_despesas,
            "indicadores": all_indicadores,
            "dados_cruzados": dados_cruzados,
            "contexto_orcamentario": contexto_orcamentario,
            "correlacoes": correlacoes,
            "anomalias": anomalias,
            "texto_analise": texto_analise,
            "message_count": counter.count,
            "data_coverage": data_coverage,
        }

        self.beliefs["result"] = result
        logger.info(
            "OrquestradorEstrela %s: pipeline complete — %d messages exchanged",
            self.agent_id,
            counter.count,
        )

        return result
