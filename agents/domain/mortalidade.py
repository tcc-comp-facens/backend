"""
Agente de Domínio — Mortalidade.

Especializado em dados de mortalidade SIM (transversal a todas as subfunções).
Consulta nós IndicadorDataSUS (tipo="mortalidade") e DespesaSIOPS de TODAS
as subfunções (301, 302, 303, 305) no Neo4j, filtrando por período e análise.

Diferente dos demais agentes de domínio, este agente NÃO filtra despesas por
uma única subfunção — ele retorna despesas de todas as subfunções porque dados
de mortalidade cruzam com todas as categorias de gasto.

Requisitos: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI, IntentionFailure

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)

# Configuração de domínio deste agente
SUBFUNCOES: list[int] = [301, 302, 303, 305]  # All subfunções (transversal)
TIPOS_INDICADOR: list[str] = ["mortalidade"]


class AgenteMortalidade(AgenteBDI):
    """Agente de domínio especializado em mortalidade (visão transversal).

    Consulta DespesaSIOPS de TODAS as subfunções (301, 302, 303, 305) e
    IndicadorDataSUS com tipo="mortalidade" no Neo4j via neo4j_client,
    filtrando por período (date_from/date_to) e análise (analysis_id).

    Diferente dos demais agentes de domínio que filtram por uma única
    subfunção, este agente mantém despesas de todas as subfunções porque
    dados de mortalidade são transversais a todas as categorias de gasto.

    Herda de AgenteBDI e implementa o ciclo BDI completo:
    perceive → deliberate → plan → execute (Req 4.4).

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
        despesas (todas as subfunções) e indicadores (mortalidade).

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

        Para "consultar_despesas": busca DespesaSIOPS e mantém registros
        de TODAS as subfunções (301, 302, 303, 305) — visão transversal (Req 4.2).

        Para "consultar_indicadores": busca IndicadorDataSUS com
        tipo="mortalidade" (Req 4.1).

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
                # Mantém despesas de TODAS as subfunções (transversal) (Req 4.2)
                despesas = [
                    d for d in all_despesas if d.get("subfuncao") in SUBFUNCOES
                ]
                self.beliefs["despesas"] = despesas
                logger.info(
                    "Agent %s: retrieved %d despesas (subfuncoes=%s)",
                    self.agent_id,
                    len(despesas),
                    SUBFUNCOES,
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
        """Recuperação de falha: retorna listas vazias (Req 4.5).

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
        """Consulta despesas (todas as subfunções) e indicadores (mortalidade).

        Método de conveniência chamado pelo orquestrador/supervisor.
        Configura as crenças, executa o ciclo BDI e retorna os dados.

        Diferente dos demais agentes de domínio, este agente retorna
        despesas de TODAS as subfunções (301, 302, 303, 305) porque
        mortalidade é transversal a todas as categorias de gasto.

        Args:
            analysis_id: ID da análise em andamento.
            date_from: Ano de início do período.
            date_to: Ano de fim do período.

        Returns:
            Dicionário com chaves "despesas" e "indicadores", cada uma
            contendo lista de registros do Neo4j. Retorna listas vazias
            se não houver dados (Req 4.5).
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
