"""
Agente de Domínio — Vigilância Epidemiológica.

Especializado em dados epidemiológicos de dengue e COVID-19 (subfunção 305).
Consulta nós IndicadorDataSUS (tipo IN ["dengue", "covid"]) e DespesaSIOPS
(subfuncao=305) no Neo4j, filtrando por período e análise.

Requisitos: 1.1, 1.2, 1.3, 1.4, 1.5
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI, IntentionFailure

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Configuração de domínio deste agente
SUBFUNCAO = 305
SUBFUNCAO_NOME = "Vigilância Epidemiológica"
TIPOS_INDICADOR: list[str] = ["dengue", "covid"]


class AgenteVigilanciaEpidemiologica(AgenteBDI):
    """Agente de domínio especializado em vigilância epidemiológica.

    Consulta DespesaSIOPS com subfunção 305 e IndicadorDataSUS com
    tipo IN ["dengue", "covid"] no Neo4j via neo4j_client, filtrando
    por período (date_from/date_to) e análise (analysis_id).

    Herda de AgenteBDI e implementa o ciclo BDI completo:
    perceive → deliberate → plan → execute (Req 1.4).

    Attributes:
        neo4j_client: Cliente Neo4j para queries Cypher.
    """

    def __init__(self, agent_id: str, neo4j_client: Neo4jClient) -> None:
        super().__init__(agent_id)
        self.neo4j_client = neo4j_client

    # ------------------------------------------------------------------
    # Ciclo BDI
    # ------------------------------------------------------------------

    def perceive(self) -> dict:
        """Percebe o ambiente a partir das crenças já definidas.

        O orquestrador/supervisor chama update_beliefs com os parâmetros
        da consulta antes de disparar o ciclo. A percepção retorna esses
        parâmetros.

        Returns:
            Dicionário com analysis_id, date_from e date_to.
        """
        return {
            "analysis_id": self.beliefs.get("analysis_id"),
            "date_from": self.beliefs.get("date_from"),
            "date_to": self.beliefs.get("date_to"),
        }

    def deliberate(self) -> list[dict]:
        """Seleciona desejos com base nas crenças atuais.

        Se os parâmetros de consulta estão presentes, deseja consultar
        despesas (subfunção 305) e indicadores (dengue, covid).

        Returns:
            Lista de desejos selecionados.
        """
        desires: list[dict] = []
        if (
            self.beliefs.get("analysis_id")
            and self.beliefs.get("date_from") is not None
        ):
            desires.append({"goal": "consultar_despesas"})
            desires.append({"goal": "consultar_indicadores"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        """Gera intenções (planos) para cada desejo.

        Args:
            desires: Lista de desejos a serem planejados.

        Returns:
            Lista de intenções com status "pending".
        """
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Executa uma intenção de consulta ao Neo4j.

        Para "consultar_despesas": busca DespesaSIOPS com subfuncao=305,
        filtrando apenas os registros da subfunção 305 (Req 1.2).

        Para "consultar_indicadores": busca IndicadorDataSUS com
        tipo IN ["dengue", "covid"] (Req 1.1).

        Args:
            intention: Intenção a ser executada.

        Raises:
            IntentionFailure: Se a consulta ao Neo4j falhar.
        """
        goal = intention["desire"]["goal"]
        analysis_id = self.beliefs["analysis_id"]
        date_from = self.beliefs["date_from"]
        date_to = self.beliefs["date_to"]

        try:
            if goal == "consultar_despesas":
                all_despesas = self.neo4j_client.get_despesas(
                    analysis_id, date_from, date_to
                )
                # Filtra apenas subfunção 305 — Vigilância Epidemiológica (Req 1.2)
                despesas = [
                    d for d in all_despesas if d.get("subfuncao") == SUBFUNCAO
                ]
                self.beliefs["despesas"] = despesas
                logger.info(
                    "Agent %s: retrieved %d despesas (subfuncao=%d)",
                    self.agent_id,
                    len(despesas),
                    SUBFUNCAO,
                )

            elif goal == "consultar_indicadores":
                indicadores = self.neo4j_client.get_indicadores(
                    analysis_id, date_from, date_to, TIPOS_INDICADOR
                )
                self.beliefs["indicadores"] = indicadores
                logger.info(
                    "Agent %s: retrieved %d indicadores (tipos=%s)",
                    self.agent_id,
                    len(indicadores),
                    TIPOS_INDICADOR,
                )

            intention["status"] = "completed"
        except Exception as e:
            raise IntentionFailure(intention, str(e)) from e

    def _recover_intention(self, failed_intention: dict) -> dict | None:
        """Recuperação de falha: retorna listas vazias (Req 1.5).

        Quando uma consulta ao Neo4j falha, o agente retorna listas
        vazias em vez de propagar a exceção, permitindo que o
        orquestrador/supervisor continue com dados parciais.

        Args:
            failed_intention: A intenção que falhou.

        Returns:
            Intenção alternativa que define listas vazias, ou None.
        """
        goal = failed_intention["desire"]["goal"]
        if goal == "consultar_despesas":
            self.beliefs["despesas"] = []
            logger.warning(
                "Agent %s: fallback — returning empty despesas", self.agent_id
            )
            return {"desire": {"goal": "noop"}, "status": "completed"}
        elif goal == "consultar_indicadores":
            self.beliefs["indicadores"] = []
            logger.warning(
                "Agent %s: fallback — returning empty indicadores",
                self.agent_id,
            )
            return {"desire": {"goal": "noop"}, "status": "completed"}
        return None

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def query(
        self,
        analysis_id: str,
        date_from: int,
        date_to: int,
    ) -> dict[str, Any]:
        """Consulta despesas (subfunção 305) e indicadores (dengue, covid).

        Método de conveniência chamado pelo orquestrador/supervisor.
        Configura as crenças, executa o ciclo BDI e retorna os dados.

        Args:
            analysis_id: ID da análise em andamento.
            date_from: Ano de início do período.
            date_to: Ano de fim do período.

        Returns:
            Dicionário com chaves "despesas" e "indicadores", cada uma
            contendo lista de registros do Neo4j. Retorna listas vazias
            se não houver dados (Req 1.5).
        """
        self.update_beliefs({
            "analysis_id": analysis_id,
            "date_from": date_from,
            "date_to": date_to,
        })

        self.run_cycle()

        return {
            "despesas": self.beliefs.get("despesas", []),
            "indicadores": self.beliefs.get("indicadores", []),
        }
