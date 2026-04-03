"""
Orquestrador central da arquitetura estrela.

Hub central que instancia e coordena AgenteConsultorStar e
AgenteAnalisadorStar, intermediando toda comunicação entre eles.
Nenhum agente periférico chama outro diretamente — toda interação
passa por este orquestrador.

Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

import logging
import uuid
from queue import Queue
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI
from agents.star.consultant import AgenteConsultorStar
from agents.star.analyzer import AgenteAnalisadorStar
from metrics import MetricsCollector

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class OrquestradorEstrela(AgenteBDI):
    """Hub central da topologia estrela (Req 5.1).

    Toda comunicação entre agentes periféricos passa por este
    orquestrador (Req 5.2). Distribui tarefas via chamadas de
    método Python (Req 5.3), registra métricas de tempo de execução
    por agente (Req 5.4), e envia evento de erro via ws_queue em
    caso de falha (Req 5.5).

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
        """Define desejos: consultar dados e depois analisar."""
        desires: list[dict] = []
        if self.beliefs.get("analysis_id"):
            desires.append({"goal": "consultar_dados"})
            desires.append({"goal": "analisar_dados"})
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
        """Executa o pipeline completo da arquitetura estrela.

        1. Instancia agentes periféricos com IDs únicos.
        2. Delega consulta ao AgenteConsultorStar (Req 5.3).
        3. Intermedia dados: passa resultado da consulta ao
           AgenteAnalisadorStar (Req 5.2).
        4. Persiste métricas de cada agente no Neo4j (Req 5.4).
        5. Em caso de falha, envia evento de erro via ws_queue
           e re-levanta a exceção (Req 5.5).

        Args:
            analysis_id: UUID da análise.
            params: Dicionário com analysis_id, date_from, date_to,
                    health_params.
            ws_queue: Fila para streaming de eventos WebSocket.

        Returns:
            Dicionário com resultado da análise.

        Raises:
            Exception: Re-levanta qualquer exceção após enviar evento
                       de erro ao ws_queue.
        """
        # Configure beliefs for BDI cycle
        self.update_beliefs({
            "analysis_id": analysis_id,
            "date_from": params.get("date_from"),
            "date_to": params.get("date_to"),
            "health_params": params.get("health_params", []),
            "ws_queue": ws_queue,
        })

        try:
            # -- 1. Instanciar agentes periféricos com IDs únicos --
            consultant_id = f"star-consultant-{uuid.uuid4().hex[:8]}"
            analyzer_id = f"star-analyzer-{uuid.uuid4().hex[:8]}"

            consultant = AgenteConsultorStar(consultant_id, self.neo4j_client)
            analyzer = AgenteAnalisadorStar(analyzer_id)

            # -- 2. Delegar consulta ao consultor (Req 5.3) --
            # Métricas do consultor (Req 5.4)
            with MetricsCollector(consultant_id, "consultor") as mc_consultant:
                data = consultant.query(
                    analysis_id=analysis_id,
                    date_from=params.get("date_from"),
                    date_to=params.get("date_to"),
                    health_params=params.get("health_params", []),
                )
            mc_consultant.persist(self.neo4j_client, analysis_id, "star")

            # -- 3. Intermediar: passar dados ao analisador (Req 5.2) --
            # Métricas do analisador (Req 5.4)
            with MetricsCollector(analyzer_id, "analisador") as mc_analyzer:
                result = analyzer.analyze(analysis_id, data, ws_queue)
            mc_analyzer.persist(self.neo4j_client, analysis_id, "star")

            # -- 4. Métricas do próprio orquestrador --
            self.beliefs["result"] = result
            return result

        except Exception as exc:
            # Req 5.5: enviar evento de erro e re-levantar
            logger.error(
                "OrquestradorEstrela %s: falha — %s", self.agent_id, exc
            )
            ws_queue.put({
                "analysisId": analysis_id,
                "architecture": "star",
                "type": "error",
                "payload": str(exc),
            })
            raise
