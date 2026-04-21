"""
Testes unitários para CoordenadorGeral (reescrito com 3 supervisores).

Valida: Requisitos 10.1, 10.5, 10.6, 10.8, 10.9, 11.1, 11.2
"""

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgenteBDI
from agents.hierarchical.coordinator import CoordenadorGeral


@pytest.fixture
def mock_neo4j():
    client = MagicMock()
    return client


@pytest.fixture
def params():
    return {
        "date_from": 2020,
        "date_to": 2022,
        "health_params": ["dengue", "covid"],
    }


@pytest.fixture
def ws_queue():
    return Queue()


class TestCoordenadorInit:
    """Req 10.1: topologia hierárquica com CoordenadorGeral."""

    def test_inherits_from_agente_bdi(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        assert isinstance(coord, AgenteBDI)

    def test_stores_neo4j_client(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        assert coord.neo4j_client is mock_neo4j

    def test_initial_state(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        assert coord.agent_id == "coord-1"
        assert coord.beliefs == {}
        assert coord.desires == []
        assert coord.intentions == []


class TestBDIOverrides:
    def test_perceive_returns_analysis_params(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        coord.update_beliefs({
            "analysis_id": "a-1",
            "date_from": 2020,
            "date_to": 2022,
            "health_params": ["dengue"],
        })
        perception = coord.perceive()
        assert perception["analysis_id"] == "a-1"
        assert perception["date_from"] == 2020
        assert perception["health_params"] == ["dengue"]

    def test_deliberate_with_analysis_id(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        coord.update_beliefs({"analysis_id": "a-1"})
        desires = coord.deliberate()
        goals = [d["goal"] for d in desires]
        assert "delegar_dominio" in goals
        assert "delegar_analitico" in goals
        assert "delegar_contexto" in goals
        assert "persistir_metricas" in goals

    def test_deliberate_without_analysis_id(self, mock_neo4j):
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        desires = coord.deliberate()
        assert desires == []


class TestRunDelegation:
    """Req 10.1, 10.5, 10.6: delegates to 3 supervisors and aggregates results."""

    def test_run_delegates_to_three_supervisors(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [{"subfuncao": 301, "ano": 2020, "valor": 1000.0}],
                "indicadores": [{"tipo": "dengue", "ano": 2020, "valor": 150.0}],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {301: {"tendencia": "crescimento"}},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [{"subfuncao": 301, "tipo": "dengue", "r": 0.8}],
                "anomalias": [],
                "texto_analise": "Análise completa.",
            }
            MockSupAnalitico.return_value._collectors = []

            result = coord.run("analysis-1", params, ws_queue)

            # SupervisorDominio was called
            MockSupDominio.return_value.run.assert_called_once()

            # SupervisorContexto was called
            MockSupContexto.return_value.run.assert_called_once()

            # SupervisorAnalitico received data via receive_from_peer (lateral)
            assert MockSupAnalitico.return_value.receive_from_peer.call_count >= 2

            # SupervisorAnalitico.run was called
            MockSupAnalitico.return_value.run.assert_called_once()

            # Result contains analysis output
            assert "correlacoes" in result
            assert "texto_analise" in result
            assert "despesas" in result
            assert "indicadores" in result
            assert "contexto_orcamentario" in result

    def test_lateral_communication_dominio_to_analitico(
        self, mock_neo4j, params, ws_queue
    ):
        """Req 10.5: SupervisorDominio → SupervisorAnalitico via receive_from_peer."""
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            dominio_data = {
                "despesas": [{"subfuncao": 301, "ano": 2020, "valor": 500.0}],
                "indicadores": [{"tipo": "covid", "ano": 2020, "valor": 200.0}],
            }
            MockSupDominio.return_value.run.return_value = dominio_data
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }
            MockSupAnalitico.return_value._collectors = []

            coord.run("analysis-1", params, ws_queue)

            # receive_from_peer was called with despesas and indicadores
            calls = MockSupAnalitico.return_value.receive_from_peer.call_args_list
            peer_data_calls = [c[0][0] for c in calls]
            # First call should have despesas and indicadores
            assert any("despesas" in d and "indicadores" in d for d in peer_data_calls)

    def test_lateral_communication_contexto_to_analitico(
        self, mock_neo4j, params, ws_queue
    ):
        """Req 10.6: SupervisorContexto → SupervisorAnalitico via receive_from_peer."""
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {301: {"tendencia": "corte"}},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }
            MockSupAnalitico.return_value._collectors = []

            coord.run("analysis-1", params, ws_queue)

            # receive_from_peer was called with contexto_orcamentario
            calls = MockSupAnalitico.return_value.receive_from_peer.call_args_list
            peer_data_calls = [c[0][0] for c in calls]
            assert any("contexto_orcamentario" in d for d in peer_data_calls)


class TestMetricsPersistence:
    """Req 10.8: registrar métricas de tempo de execução por agente e supervisor."""

    def test_persists_metrics_for_supervisors(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }
            MockSupAnalitico.return_value._collectors = []

            coord.run("analysis-1", params, ws_queue)

            # save_metrica should be called 3 times: once per supervisor
            assert mock_neo4j.save_metrica.call_count == 3

            # All calls should use architecture "hierarchical"
            for call in mock_neo4j.save_metrica.call_args_list:
                metrica = call[0][0]
                analysis = call[0][1]
                assert analysis == "analysis-1"
                assert metrica["architecture"] == "hierarchical"
                assert "executionTimeMs" in metrica
                assert "cpuPercent" in metrica
                assert "memoryMb" in metrica


class TestGracefulDegradation:
    """Req 10.9: degradação graciosa quando supervisor falha."""

    def test_continues_when_dominio_fails(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.side_effect = RuntimeError(
                "Neo4j connection failed"
            )
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "Análise parcial.",
            }
            MockSupAnalitico.return_value._collectors = []

            # Should NOT raise — graceful degradation
            result = coord.run("analysis-1", params, ws_queue)

            # Error event should be in the queue
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) >= 1
            assert "SupervisorDominio" in error_events[0]["payload"]
            assert error_events[0]["architecture"] == "hierarchical"

            # Analysis still ran with empty data
            assert "texto_analise" in result

    def test_continues_when_analitico_fails(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [{"subfuncao": 301}],
                "indicadores": [],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.side_effect = ValueError(
                "LLM timeout"
            )
            MockSupAnalitico.return_value._collectors = []

            # Should NOT raise — graceful degradation
            result = coord.run("analysis-1", params, ws_queue)

            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) >= 1
            assert "SupervisorAnalitico" in error_events[0]["payload"]

            # Result has default empty values
            assert result["correlacoes"] == []
            assert result["anomalias"] == []
            assert result["texto_analise"] == ""

    def test_all_supervisors_fail_returns_empty_result(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.side_effect = RuntimeError("fail1")
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.side_effect = RuntimeError("fail2")
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.side_effect = RuntimeError("fail3")
            MockSupAnalitico.return_value._collectors = []

            result = coord.run("analysis-1", params, ws_queue)

            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) == 3

            assert result["correlacoes"] == []
            assert result["texto_analise"] == ""

    def test_metrics_persist_failure_does_not_crash(
        self, mock_neo4j, params, ws_queue
    ):
        """Metrics persistence failure should be logged but not crash."""
        coord = CoordenadorGeral("coord-1", mock_neo4j)
        mock_neo4j.save_metrica.side_effect = RuntimeError("DB down")

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }
            MockSupAnalitico.return_value._collectors = []

            # Should NOT raise even though metrics persistence fails
            result = coord.run("analysis-1", params, ws_queue)
            assert "correlacoes" in result


class TestMessageCount:
    """Req 11.1, 11.2: message counter integration."""

    def test_message_count_in_result(
        self, mock_neo4j, params, ws_queue
    ):
        coord = CoordenadorGeral("coord-1", mock_neo4j)

        with patch(
            "agents.hierarchical.coordinator.SupervisorDominio"
        ) as MockSupDominio, patch(
            "agents.hierarchical.coordinator.SupervisorAnalitico"
        ) as MockSupAnalitico, patch(
            "agents.hierarchical.coordinator.SupervisorContexto"
        ) as MockSupContexto:
            MockSupDominio.return_value.run.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockSupDominio.return_value._collectors = []
            MockSupContexto.return_value.run.return_value = {
                "contexto_orcamentario": {},
            }
            MockSupContexto.return_value._collectors = []
            MockSupAnalitico.return_value.run.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }
            MockSupAnalitico.return_value._collectors = []

            result = coord.run("analysis-1", params, ws_queue)

            # message_count should be present and > 0
            assert "message_count" in result
            assert result["message_count"] > 0

            # metric event should be in the queue
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            metric_events = [e for e in events if e.get("type") == "metric"]
            assert len(metric_events) >= 1


class TestExportsFromInit:
    """Verify __init__.py exports CoordenadorGeral and new supervisors."""

    def test_import_coordenador_geral(self):
        from agents.hierarchical import CoordenadorGeral as CG
        assert CG is not None

    def test_import_all_hierarchical_agents(self):
        from agents.hierarchical import (
            CoordenadorGeral,
            SupervisorDominio,
            SupervisorAnalitico,
            SupervisorContexto,
        )
        assert CoordenadorGeral is not None
        assert SupervisorDominio is not None
        assert SupervisorAnalitico is not None
        assert SupervisorContexto is not None
