"""
Unit tests for AgenteMortalidade.

Validates Requirements 4.1, 4.2, 4.3, 4.4, 4.5.
"""

from unittest.mock import MagicMock

from agents.base import AgenteBDI
from agents.domain.mortalidade import (
    AgenteMortalidade,
    SUBFUNCOES,
    TIPOS_INDICADOR,
)


class TestInheritanceAndInit:
    """Req 4.4: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        assert isinstance(agent, AgenteBDI)

    def test_stores_neo4j_client(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        assert agent.neo4j_client is client

    def test_agent_id_set(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        assert agent.agent_id == "mo-test-1"

    def test_initial_state_empty(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []


class TestBDICycle:
    """Req 4.4: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        agent.update_beliefs({
            "analysis_id": "a1",
            "date_from": 2019,
            "date_to": 2023,
        })
        perception = agent.perceive()
        assert perception["analysis_id"] == "a1"
        assert perception["date_from"] == 2019
        assert perception["date_to"] == 2023

    def test_deliberate_creates_two_desires(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        agent.update_beliefs({
            "analysis_id": "a1",
            "date_from": 2019,
            "date_to": 2023,
        })
        desires = agent.deliberate()
        goals = [d["goal"] for d in desires]
        assert "consultar_despesas" in goals
        assert "consultar_indicadores" in goals

    def test_deliberate_no_desires_without_params(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        client = MagicMock()
        agent = AgenteMortalidade("mo-test-1", client)
        desires = [{"goal": "consultar_despesas"}, {"goal": "consultar_indicadores"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 2
        assert all(i["status"] == "pending" for i in intentions)


class TestQueryTransversal:
    """Req 4.1, 4.2, 4.3: Query returns ALL subfunções and tipo="mortalidade"."""

    def test_query_returns_despesas_from_all_subfuncoes(self):
        """Req 4.2: Returns despesas from ALL subfunções (301, 302, 303, 305)."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 200.0},
            {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2020, "valor": 150.0},
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 250.0},
        ]
        client.get_indicadores.return_value = [
            {"tipo": "mortalidade", "ano": 2020, "valor": 42.0},
        ]

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["despesas"]) == 4
        subfuncoes_returned = {d["subfuncao"] for d in result["despesas"]}
        assert subfuncoes_returned == {301, 302, 303, 305}

    def test_query_filters_out_unknown_subfuncoes(self):
        """Only subfunções in SUBFUNCOES (301, 302, 303, 305) are kept."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 999, "subfuncaoNome": "Desconhecida", "ano": 2020, "valor": 50.0},
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 250.0},
        ]
        client.get_indicadores.return_value = []

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["despesas"]) == 2
        subfuncoes_returned = {d["subfuncao"] for d in result["despesas"]}
        assert 999 not in subfuncoes_returned

    def test_query_returns_mortalidade_indicadores(self):
        """Req 4.1: Queries IndicadorDataSUS tipo="mortalidade"."""
        client = MagicMock()
        client.get_despesas.return_value = []
        client.get_indicadores.return_value = [
            {"tipo": "mortalidade", "ano": 2020, "valor": 42.0},
            {"tipo": "mortalidade", "ano": 2021, "valor": 38.0},
        ]

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["indicadores"]) == 2
        # Verify get_indicadores was called with the correct tipos
        client.get_indicadores.assert_called_once_with(
            "a1", 2019, 2023, TIPOS_INDICADOR
        )

    def test_query_returns_required_structure(self):
        """Req 4.3: Returns dict with despesas and indicadores keys."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 200.0},
        ]
        client.get_indicadores.return_value = [
            {"tipo": "mortalidade", "ano": 2020, "valor": 42.0},
        ]

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert "despesas" in result
        assert "indicadores" in result
        assert isinstance(result["despesas"], list)
        assert isinstance(result["indicadores"], list)

        # Verify required fields in despesa records
        for d in result["despesas"]:
            assert "subfuncao" in d
            assert "ano" in d
            assert "valor" in d

        # Verify required fields in indicador records
        for i in result["indicadores"]:
            assert "tipo" in i
            assert "ano" in i
            assert "valor" in i

    def test_query_preserves_multiple_years_per_subfuncao(self):
        """Transversal agent keeps all years for all subfunções."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2019, "valor": 90.0},
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2019, "valor": 180.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 200.0},
        ]
        client.get_indicadores.return_value = []

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["despesas"]) == 4


class TestEmptyData:
    """Req 4.5: Returns empty lists without exception when no data found."""

    def test_empty_neo4j_returns_empty_lists(self):
        client = MagicMock()
        client.get_despesas.return_value = []
        client.get_indicadores.return_value = []

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert result["despesas"] == []
        assert result["indicadores"] == []

    def test_only_unknown_subfuncoes_returns_empty_despesas(self):
        """When Neo4j returns despesas but none match known subfunções."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 999, "subfuncaoNome": "Desconhecida", "ano": 2020, "valor": 50.0},
            {"subfuncao": 888, "subfuncaoNome": "Outra", "ano": 2020, "valor": 75.0},
        ]
        client.get_indicadores.return_value = []

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert result["despesas"] == []

    def test_neo4j_failure_returns_empty_lists(self):
        """Req 4.5: Graceful degradation on Neo4j failure."""
        client = MagicMock()
        client.get_despesas.side_effect = Exception("Neo4j connection failed")
        client.get_indicadores.side_effect = Exception("Neo4j connection failed")

        agent = AgenteMortalidade("mo-test-1", client)
        result = agent.query("a1", 2019, 2023)

        # Recovery should produce empty lists, not raise
        assert result["despesas"] == []
        assert result["indicadores"] == []


class TestDomainConstants:
    """Verify domain configuration constants."""

    def test_subfuncoes_all(self):
        assert set(SUBFUNCOES) == {301, 302, 303, 305}

    def test_subfuncoes_is_list(self):
        assert isinstance(SUBFUNCOES, list)

    def test_tipos_indicador(self):
        assert set(TIPOS_INDICADOR) == {"mortalidade"}
