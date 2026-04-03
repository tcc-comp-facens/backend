"""
Coletor de métricas de execução por agente.

Usa psutil para capturar CPU e memória, e time para medir duração.
Cada agente instancia um MetricsCollector no início do seu trabalho
e chama persist() ao finalizar para salvar um nó MetricaExecucao no Neo4j.

Requisitos: 11.1, 11.2, 11.3, 11.4
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient


class MetricsCollector:
    """Coleta métricas de execução (tempo, CPU, memória) para um agente.

    Pode ser usado como context manager::

        with MetricsCollector("agent-1", "consultor") as mc:
            # ... trabalho do agente ...
        mc.persist(neo4j_client, analysis_id, "star")

    Ou com start/stop explícitos::

        mc = MetricsCollector("agent-1", "consultor")
        mc.start()
        # ... trabalho do agente ...
        mc.stop()
        mc.persist(neo4j_client, analysis_id, "star")
    """

    def __init__(self, agent_id: str, agent_type: str) -> None:
        self.agent_id = agent_id
        self.agent_type = agent_type
        self._process = psutil.Process()
        self._start_time: float | None = None
        self._end_time: float | None = None
        # Warm-up cpu_percent so the first real call returns a meaningful value.
        self._process.cpu_percent()

    # -- context manager ------------------------------------------------

    def __enter__(self) -> "MetricsCollector":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.stop()

    # -- start / stop ---------------------------------------------------

    def start(self) -> None:
        """Marca o início da medição."""
        self._start_time = time.time()

    def stop(self) -> None:
        """Marca o fim da medição."""
        self._end_time = time.time()

    # -- collect --------------------------------------------------------

    def collect(self) -> dict:
        """Retorna as métricas coletadas como dicionário.

        Returns:
            Dict com agentId, agentType, executionTimeMs, cpuPercent e memoryMb.

        Raises:
            RuntimeError: Se start() não foi chamado.
        """
        if self._start_time is None:
            raise RuntimeError("MetricsCollector.start() must be called before collect()")

        end = self._end_time if self._end_time is not None else time.time()

        return {
            "agentId": self.agent_id,
            "agentType": self.agent_type,
            "executionTimeMs": int((end - self._start_time) * 1000),
            "cpuPercent": self._process.cpu_percent(),
            "memoryMb": round(self._process.memory_info().rss / (1024 * 1024), 2),
        }

    # -- persist --------------------------------------------------------

    def persist(
        self,
        neo4j_client: "Neo4jClient",
        analysis_id: str,
        architecture: str,
    ) -> dict:
        """Coleta métricas e persiste um nó MetricaExecucao no Neo4j.

        Args:
            neo4j_client: Cliente Neo4j com método save_metrica.
            analysis_id: UUID da análise em andamento.
            architecture: "star" ou "hierarchical".

        Returns:
            O dicionário de métricas persistido.
        """
        metrics = self.collect()
        metrics["id"] = str(uuid.uuid4())
        metrics["architecture"] = architecture
        metrics["recordedAt"] = datetime.now(timezone.utc).isoformat()

        neo4j_client.save_metrica(metrics, analysis_id)
        return metrics
