"""
Unit tests for AgenteVigilanciaEpidemiologica.

Validates Requirements 1.1, 1.2, 1.3, 1.4, 1.5.
"""

from unittest.mock import MagicMock

from agents.base import AgenteBDI
from agents.domain.vigilancia_epidemiologica import (
    AgenteVigilanciaEpidemiologica,
    SUBFUNCAO,
    TIPOS_INDICADOR,
)


class TestInheritanceAndInit:
    """Req 1.4: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        assert isinstance(agent, AgenteBDI)

    def test_stores_neo4j_client(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        assert agent.neo4j_client is client

    def test_agent_id_set(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        assert agent.agent_id == "ve-test-1"

    def test_initial_state_empty(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []


class TestBDICycle:
    """Req 1.4: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
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
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
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
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        client = MagicMock()
        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        desires = [{"goal": "consultar_despesas"}, {"goal": "consultar_indicadores"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 2
        assert all(i["status"] == "pending" for i in intentions)


class TestQuery:
    """Req 1.1, 1.2, 1.3: Query filters by subfuncao=305 and tipo IN [dengue, covid]."""

    def test_query_filters_despesas_subfuncao_305(self):
        """Req 1.2: Only subfuncao 305 despesas are returned."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 200.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 300.0},
        ]
        client.get_indicadores.return_value = [
            {"tipo": "dengue", "ano": 2020, "valor": 50.0},
        ]

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["despesas"]) == 1
        assert result["despesas"][0]["subfuncao"] == 305

    def test_query_returns_dengue_and_covid_indicadores(self):
        """Req 1.1: Queries IndicadorDataSUS tipo IN [dengue, covid]."""
        client = MagicMock()
        client.get_despesas.return_value = []
        client.get_indicadores.return_value = [
            {"tipo": "dengue", "ano": 2020, "valor": 50.0},
            {"tipo": "covid", "ano": 2021, "valor": 75.0},
        ]

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert len(result["indicadores"]) == 2
        # Verify get_indicadores was called with the correct tipos
        client.get_indicadores.assert_called_once_with(
            "a1", 2019, 2023, TIPOS_INDICADOR
        )

    def test_query_returns_required_structure(self):
        """Req 1.3: Returns dict with despesas and indicadores keys."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 200.0},
        ]
        client.get_indicadores.return_value = [
            {"tipo": "dengue", "ano": 2020, "valor": 50.0},
        ]

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
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


class TestEmptyData:
    """Req 1.5: Returns empty lists without exception when no data found."""

    def test_empty_neo4j_returns_empty_lists(self):
        client = MagicMock()
        client.get_despesas.return_value = []
        client.get_indicadores.return_value = []

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert result["despesas"] == []
        assert result["indicadores"] == []

    def test_no_matching_subfuncao_returns_empty_despesas(self):
        """When Neo4j returns despesas but none match subfuncao 305."""
        client = MagicMock()
        client.get_despesas.return_value = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 100.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 300.0},
        ]
        client.get_indicadores.return_value = []

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        result = agent.query("a1", 2019, 2023)

        assert result["despesas"] == []

    def test_neo4j_failure_returns_empty_lists(self):
        """Req 1.5: Graceful degradation on Neo4j failure."""
        client = MagicMock()
        client.get_despesas.side_effect = Exception("Neo4j connection failed")
        client.get_indicadores.side_effect = Exception("Neo4j connection failed")

        agent = AgenteVigilanciaEpidemiologica("ve-test-1", client)
        result = agent.query("a1", 2019, 2023)

        # Recovery should produce empty lists, not raise
        assert result["despesas"] == []
        assert result["indicadores"] == []


class TestDomainConstants:
    """Verify domain configuration constants."""

    def test_subfuncao_is_305(self):
        assert SUBFUNCAO == 305

    def test_tipos_indicador(self):
        assert set(TIPOS_INDICADOR) == {"dengue", "covid"}
