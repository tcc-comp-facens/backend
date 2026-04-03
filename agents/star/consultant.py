"""
Agente Consultor da arquitetura estrela.

Consulta dados financeiros (DespesaSIOPS) e indicadores de saúde
(IndicadorDataSUS) exclusivamente do Neo4j local, filtrando por
período e parâmetros de saúde. Não faz chamadas a APIs externas.

Requisitos: 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI, IntentionFailure

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class AgenteConsultorStar(AgenteBDI):
    """Agente consultor da topologia estrela.

    Consulta DespesaSIOPS e IndicadorDataSUS no Neo4j via neo4j_client,
    filtrando por período (date_from/date_to) e parâmetros de saúde
    (dengue, covid, vacinação). Toda comunicação passa pelo
    OrquestradorEstrela — este agente não chama outros agentes.

    Attributes:
        neo4j_client: Cliente Neo4j para queries Cypher.
    """

    def __init__(self, agent_id: str, neo4j_client: Neo4jClient):
        super().__init__(agent_id)
        self.neo4j_client = neo4j_client

    def perceive(self) -> dict:
        """Percebe o ambiente a partir das crenças já definidas pelo orquestrador.

        O orquestrador chama update_beliefs com os parâmetros da consulta
        antes de disparar o ciclo. A percepção retorna esses parâmetros.
        """
        return {
            "analysis_id": self.beliefs.get("analysis_id"),
            "date_from": self.beliefs.get("date_from"),
            "date_to": self.beliefs.get("date_to"),
            "health_params": self.beliefs.get("health_params", []),
        }

    def deliberate(self) -> list[dict]:
        """Seleciona desejos com base nas crenças atuais.

        Se os parâmetros de consulta estão presentes, deseja consultar
        despesas e indicadores.
        """
        desires: list[dict] = []
        if self.beliefs.get("analysis_id") and self.beliefs.get("date_from") is not None:
            desires.append({"goal": "consultar_despesas"})
            if self.beliefs.get("health_params"):
                desires.append({"goal": "consultar_indicadores"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Gera intenções (planos) para cada desejo."""
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Executa uma intenção de consulta ao Neo4j."""
        goal = intention["desire"]["goal"]
        analysis_id = self.beliefs["analysis_id"]
        date_from = self.beliefs["date_from"]
        date_to = self.beliefs["date_to"]

        try:
            if goal == "consultar_despesas":
                data = self.neo4j_client.get_despesas(analysis_id, date_from, date_to)
                self.beliefs["despesas"] = data
                logger.info(
                    "Agent %s: retrieved %d despesas", self.agent_id, len(data)
                )
            elif goal == "consultar_indicadores":
                health_params = self.beliefs.get("health_params", [])
                data = self.neo4j_client.get_indicadores(
                    analysis_id, date_from, date_to, health_params
                )
                self.beliefs["indicadores"] = data
                logger.info(
                    "Agent %s: retrieved %d indicadores", self.agent_id, len(data)
                )
            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    def query(
        self,
        analysis_id: str,
        date_from: int,
        date_to: int,
        health_params: list[str],
    ) -> dict[str, Any]:
        """Consulta despesas e indicadores no Neo4j.

        Método de conveniência chamado pelo OrquestradorEstrela.
        Configura as crenças, executa o ciclo BDI e retorna os dados.

        Args:
            analysis_id: ID da análise em andamento.
            date_from: Ano de início do período.
            date_to: Ano de fim do período.
            health_params: Lista de parâmetros de saúde (ex: ["dengue", "covid"]).

        Returns:
            Dicionário com chaves "despesas" e "indicadores", cada uma
            contendo lista de registros do Neo4j.
        """
        self.update_beliefs({
            "analysis_id": analysis_id,
            "date_from": date_from,
            "date_to": date_to,
            "health_params": health_params,
        })

        self.run_cycle()

        return {
            "despesas": self.beliefs.get("despesas", []),
            "indicadores": self.beliefs.get("indicadores", []),
        }
