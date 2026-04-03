# Feature: multiagent-architecture-comparison, Property 17: Registro completo de métricas
"""
Property-based tests for MetricsCollector — complete metrics registration.

**Validates: Requirements 5.4, 6.4, 11.1, 11.2, 11.3, 11.4**

Verifies that for any completed analysis, MetricaExecucao nodes contain all
required fields (id, architecture, agentId, agentType, executionTimeMs,
cpuPercent, memoryMb, recordedAt) for each participating agent.
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from metrics import MetricsCollector

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_ARCHITECTURES = ["star", "hierarchical"]

VALID_AGENT_TYPES = [
    "consultor",
    "analisador",
    "orquestrador",
    "coordenador",
    "supervisor_consulta",
    "supervisor_analise",
]

# Agents that participate in each architecture
STAR_AGENTS = [
    ("orquestrador-star", "orquestrador"),
    ("consultor-star", "consultor"),
    ("analisador-star", "analisador"),
]

HIER_AGENTS = [
    ("coordenador-hier", "coordenador"),
    ("supervisor-consulta-hier", "supervisor_consulta"),
    ("supervisor-analise-hier", "supervisor_analise"),
    ("consultor-hier", "consultor"),
    ("analisador-hier", "analisador"),
]

REQUIRED_METRICA_FIELDS = {
    "id",
    "architecture",
    "agentId",
    "agentType",
    "executionTimeMs",
    "cpuPercent",
    "memoryMb",
    "recordedAt",
}

agent_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "Nd", "Pd")),
    min_size=1,
    max_size=30,
)

agent_type_strategy = st.sampled_from(VALID_AGENT_TYPES)

architecture_strategy = st.sampled_from(VALID_ARCHITECTURES)

analysis_id_strategy = st.from_type(uuid.UUID).map(str)


# ---------------------------------------------------------------------------
# P17 — Registro completo de métricas
# ---------------------------------------------------------------------------


class TestP17RegistroCompletoDeMetricas:
    """Property test P17: For any completed analysis, MetricaExecucao nodes
    must exist with ALL required fields for each participating agent.

    **Validates: Requirements 5.4, 6.4, 11.1, 11.2, 11.3, 11.4**
    """

    @given(
        agent_id=agent_id_strategy,
        agent_type=agent_type_strategy,
        architecture=architecture_strategy,
        analysis_id=analysis_id_strategy,
    )
    @settings(max_examples=100)
    def test_p17_persist_produces_all_required_fields(
        self, agent_id, agent_type, architecture, analysis_id
    ):
        """For any agent identity and architecture, persist() must produce
        a MetricaExecucao dict containing every required field."""
        mock_client = MagicMock()

        mc = MetricsCollector(agent_id, agent_type)
        mc.start()
        mc.stop()

        result = mc.persist(mock_client, analysis_id, architecture)

        missing = REQUIRED_METRICA_FIELDS - set(result.keys())
        assert not missing, f"Missing required fields: {missing}"

    @given(
        agent_id=agent_id_strategy,
        agent_type=agent_type_strategy,
        architecture=architecture_strategy,
        analysis_id=analysis_id_strategy,
    )
    @settings(max_examples=100)
    def test_p17_persist_fields_have_correct_types(
        self, agent_id, agent_type, architecture, analysis_id
    ):
        """For any agent, persisted metrics must have correct types:
        - id: non-empty str (UUID)
        - architecture: 'star' or 'hierarchical'
        - agentId: non-empty str
        - agentType: non-empty str
        - executionTimeMs: int >= 0
        - cpuPercent: float >= 0
        - memoryMb: float > 0
        - recordedAt: non-empty str (ISO datetime)
        """
        mock_client = MagicMock()

        mc = MetricsCollector(agent_id, agent_type)
        mc.start()
        mc.stop()

        m = mc.persist(mock_client, analysis_id, architecture)

        assert isinstance(m["id"], str) and len(m["id"]) > 0
        assert m["architecture"] in VALID_ARCHITECTURES
        assert isinstance(m["agentId"], str) and len(m["agentId"]) > 0
        assert isinstance(m["agentType"], str) and len(m["agentType"]) > 0
        assert isinstance(m["executionTimeMs"], int) and m["executionTimeMs"] >= 0
        assert isinstance(m["cpuPercent"], float)
        assert isinstance(m["memoryMb"], float) and m["memoryMb"] > 0
        assert isinstance(m["recordedAt"], str) and len(m["recordedAt"]) > 0

    @given(
        agent_id=agent_id_strategy,
        agent_type=agent_type_strategy,
        architecture=architecture_strategy,
        analysis_id=analysis_id_strategy,
    )
    @settings(max_examples=100)
    def test_p17_persist_calls_save_metrica_with_analysis_id(
        self, agent_id, agent_type, architecture, analysis_id
    ):
        """persist() must call save_metrica on the Neo4j client with the
        correct analysis_id so the GEROU_METRICA relationship is created."""
        mock_client = MagicMock()

        mc = MetricsCollector(agent_id, agent_type)
        mc.start()
        mc.stop()
        mc.persist(mock_client, analysis_id, architecture)

        mock_client.save_metrica.assert_called_once()
        call_args = mock_client.save_metrica.call_args[0]
        assert call_args[1] == analysis_id

    @given(analysis_id=analysis_id_strategy)
    @settings(max_examples=100)
    def test_p17_star_architecture_registers_all_agents(self, analysis_id):
        """For any star architecture analysis, every participating agent
        (orquestrador, consultor, analisador) must produce a complete
        MetricaExecucao record."""
        mock_client = MagicMock()
        persisted = []

        for aid, atype in STAR_AGENTS:
            mc = MetricsCollector(aid, atype)
            mc.start()
            mc.stop()
            result = mc.persist(mock_client, analysis_id, "star")
            persisted.append(result)

        assert len(persisted) == len(STAR_AGENTS)

        for metrica in persisted:
            missing = REQUIRED_METRICA_FIELDS - set(metrica.keys())
            assert not missing, f"Missing fields in star agent metric: {missing}"
            assert metrica["architecture"] == "star"

        registered_agents = {m["agentId"] for m in persisted}
        expected_agents = {aid for aid, _ in STAR_AGENTS}
        assert registered_agents == expected_agents

    @given(analysis_id=analysis_id_strategy)
    @settings(max_examples=100)
    def test_p17_hierarchical_architecture_registers_all_agents(self, analysis_id):
        """For any hierarchical architecture analysis, every participating
        agent (coordenador, supervisors, consultor, analisador) must produce
        a complete MetricaExecucao record."""
        mock_client = MagicMock()
        persisted = []

        for aid, atype in HIER_AGENTS:
            mc = MetricsCollector(aid, atype)
            mc.start()
            mc.stop()
            result = mc.persist(mock_client, analysis_id, "hierarchical")
            persisted.append(result)

        assert len(persisted) == len(HIER_AGENTS)

        for metrica in persisted:
            missing = REQUIRED_METRICA_FIELDS - set(metrica.keys())
            assert not missing, f"Missing fields in hier agent metric: {missing}"
            assert metrica["architecture"] == "hierarchical"

        registered_agents = {m["agentId"] for m in persisted}
        expected_agents = {aid for aid, _ in HIER_AGENTS}
        assert registered_agents == expected_agents

    @given(
        agent_id=agent_id_strategy,
        agent_type=agent_type_strategy,
        analysis_id=analysis_id_strategy,
    )
    @settings(max_examples=100)
    def test_p17_each_persist_generates_unique_id(
        self, agent_id, agent_type, analysis_id
    ):
        """Each call to persist() must generate a unique metric id,
        ensuring no MetricaExecucao nodes collide."""
        mock_client = MagicMock()

        mc = MetricsCollector(agent_id, agent_type)
        mc.start()
        mc.stop()

        r1 = mc.persist(mock_client, analysis_id, "star")
        r2 = mc.persist(mock_client, analysis_id, "star")

        assert r1["id"] != r2["id"], "Each persist must generate a unique id"
