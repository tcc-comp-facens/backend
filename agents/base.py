"""
Classe base AgenteBDI — modelo Belief-Desire-Intention.

Todos os agentes do sistema (estrela e hierárquico) herdam desta classe.
Implementa o ciclo BDI completo: perceive → update_beliefs → deliberate → plan → execute.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class IntentionFailure(Exception):
    """Raised when an intention fails during execution."""

    def __init__(self, intention: dict, reason: str):
        self.intention = intention
        self.reason = reason
        super().__init__(f"Intention failed: {reason}")


class AgenteBDI:
    """Classe base para todos os agentes BDI.

    Attributes:
        agent_id: Identificador único do agente.
        beliefs: Base de crenças sobre o estado do ambiente (Req 7.1).
        desires: Objetivos a serem alcançados (Req 7.2).
        intentions: Planos de ação selecionados para atingir os desejos (Req 7.3).
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.beliefs: dict[str, Any] = {}
        self.desires: list[dict] = []
        self.intentions: list[dict] = []
        self._failed_intentions: list[dict] = []

    def perceive(self) -> dict:
        """Percebe o ambiente (consulta Neo4j ou recebe dados do orquestrador).

        Subclasses devem sobrescrever este método para implementar
        a percepção específica do agente.

        Returns:
            Dicionário com as percepções do ambiente.
        """
        return {}

    def update_beliefs(self, perception: dict) -> None:
        """Atualiza crenças com base na percepção (Req 7.4).

        Args:
            perception: Dicionário com novas informações do ambiente.
        """
        self.beliefs.update(perception)

    def deliberate(self) -> list[dict]:
        """Seleciona desejos alcançáveis dado as crenças atuais.

        Subclasses devem sobrescrever para implementar lógica de
        deliberação específica. A implementação base retorna todos
        os desejos como alcançáveis.

        Returns:
            Lista de desejos selecionados.
        """
        return list(self.desires)

    def plan(self, desires: list[dict]) -> list[dict]:
        """Gera planos (intenções) para atingir os desejos.

        Subclasses devem sobrescrever para implementar lógica de
        planejamento específica. A implementação base converte cada
        desejo em uma intenção direta.

        Args:
            desires: Lista de desejos a serem planejados.

        Returns:
            Lista de intenções (planos de ação).
        """
        return [{"desire": d, "status": "pending"} for d in desires]

    def _execute_intention(self, intention: dict) -> None:
        """Executa uma única intenção.

        Subclasses devem sobrescrever para implementar a execução real.

        Args:
            intention: Intenção a ser executada.

        Raises:
            IntentionFailure: Se a intenção falhar.
        """
        intention["status"] = "completed"

    def execute(self, intentions: list[dict]) -> None:
        """Executa as intenções selecionadas com recuperação de falha (Req 7.5).

        Quando uma intenção falha, tenta selecionar uma alternativa.
        Se não houver alternativa, reporta a falha — nunca permanece
        em estado indefinido.

        Args:
            intentions: Lista de intenções a executar.
        """
        for intention in intentions:
            try:
                self._execute_intention(intention)
            except IntentionFailure as e:
                logger.warning(
                    "Agent %s: intention failed — %s", self.agent_id, e.reason
                )
                intention["status"] = "failed"
                self._failed_intentions.append(intention)

                alternative = self._recover_intention(intention)
                if alternative is not None:
                    logger.info(
                        "Agent %s: trying alternative intention", self.agent_id
                    )
                    try:
                        self._execute_intention(alternative)
                    except IntentionFailure as e2:
                        logger.error(
                            "Agent %s: alternative also failed — %s",
                            self.agent_id,
                            e2.reason,
                        )
                        alternative["status"] = "failed"
                        self._failed_intentions.append(alternative)
                else:
                    logger.error(
                        "Agent %s: no alternative available, reporting failure",
                        self.agent_id,
                    )

    def _recover_intention(self, failed_intention: dict) -> dict | None:
        """Tenta encontrar uma intenção alternativa após falha (Req 7.5).

        Subclasses podem sobrescrever para implementar estratégias de
        recuperação mais sofisticadas.

        Args:
            failed_intention: A intenção que falhou.

        Returns:
            Uma intenção alternativa, ou None se não houver alternativa.
        """
        return None

    def run_cycle(self) -> None:
        """Ciclo BDI completo: perceive → update_beliefs → deliberate → plan → execute."""
        perception = self.perceive()
        self.update_beliefs(perception)
        desires = self.deliberate()
        self.intentions = self.plan(desires)
        self.execute(self.intentions)
