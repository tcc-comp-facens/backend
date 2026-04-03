"""
Testes unitários para OrquestradorEstrela.

Valida: Requisitos 5.1, 5.2, 5.3, 5.4, 5.5
"""

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgenteBDI
from agents.star.orchestrator import OrquestradorEstrela


@pytest.fixture
def mock_neo4j():
    client = MagicMock()
    client.get_despesas.return_value = [
        {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 1000.0},
    ]
    client.get_indicadores.return_value = [
        {"tipo": "dengue", "ano": 2020, "valor": 150.0},
    ]
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


class TestOrquestradorInit:
    """Req 5.1: topologia estrela com OrquestradorEstrela central."""

    def test_inherits_from_agente_bdi(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        assert isinstance(orch, AgenteBDI)

    def test_stores_neo4j_client(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        assert orch.neo4j_client is mock_neo4j

    def test_initial_state(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        assert orch.agent_id == "orch-1"
        assert orch.beliefs == {}
        assert orch.desires == []
        assert orch.intentions == []


class TestBDIOverrides:
    def test_perceive_returns_analysis_params(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        orch.update_beliefs({
            "analysis_id": "a-1",
            "date_from": 2020,
            "date_to": 2022,
            "health_params": ["dengue"],
        })
        perception = orch.perceive()
        assert perception["analysis_id"] == "a-1"
        assert perception["date_from"] == 2020
        assert perception["health_params"] == ["dengue"]

    def test_deliberate_with_analysis_id(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        orch.update_beliefs({"analysis_id": "a-1"})
        desires = orch.deliberate()
        goals = [d["goal"] for d in desires]
        assert "consultar_dados" in goals
        assert "analisar_dados" in goals
        assert "persistir_metricas" in goals

    def test_deliberate_without_analysis_id(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        desires = orch.deliberate()
        assert desires == []


class TestRunIntermediatesCommunication:
    """Req 5.2: OrquestradorEstrela intermedia toda comunicação."""

    def test_run_delegates_to_consultant_then_analyzer(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)

        with patch(
            "agents.star.orchestrator.AgenteConsultorStar"
        ) as MockConsultant, patch(
            "agents.star.orchestrator.AgenteAnalisadorStar"
        ) as MockAnalyzer:
            mock_consultant = MockConsultant.return_value
            mock_consultant.query.return_value = {
                "despesas": [{"subfuncao": 301, "ano": 2020, "valor": 1000.0}],
                "indicadores": [{"tipo": "dengue", "ano": 2020, "valor": 150.0}],
            }
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "Análise completa.",
            }

            result = orch.run("analysis-1", params, ws_queue)

            # Consultant was called with correct params (Req 5.3)
            mock_consultant.query.assert_called_once_with(
                analysis_id="analysis-1",
                date_from=2020,
                date_to=2022,
                health_params=["dengue", "covid"],
            )

            # Analyzer received data FROM the consultant via orchestrator (Req 5.2)
            mock_analyzer.analyze.assert_called_once()
            call_args = mock_analyzer.analyze.call_args
            assert call_args[0][0] == "analysis-1"
            assert call_args[0][1] == mock_consultant.query.return_value
            assert call_args[0][2] is ws_queue

            assert "correlacoes" in result
            assert "texto_analise" in result


class TestMetricsPersistence:
    """Req 5.4: registrar métricas de tempo de execução por agente."""

    def test_persists_metrics_for_both_agents(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)

        with patch(
            "agents.star.orchestrator.AgenteConsultorStar"
        ) as MockConsultant, patch(
            "agents.star.orchestrator.AgenteAnalisadorStar"
        ) as MockAnalyzer:
            MockConsultant.return_value.query.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockAnalyzer.return_value.analyze.return_value = {
                "correlacoes": [],
                "anomalias": [],
                "texto_analise": "",
            }

            orch.run("analysis-1", params, ws_queue)

            # save_metrica should be called twice: once for consultant, once for analyzer
            assert mock_neo4j.save_metrica.call_count == 2

            # Both calls should use architecture "star"
            for call in mock_neo4j.save_metrica.call_args_list:
                metrica = call[0][0]
                analysis = call[0][1]
                assert analysis == "analysis-1"
                assert metrica["architecture"] == "star"
                assert "executionTimeMs" in metrica
                assert "cpuPercent" in metrica
                assert "memoryMb" in metrica


class TestErrorHandling:
    """Req 5.5: enviar evento de erro via WebSocket e encerrar."""

    def test_sends_error_event_on_consultant_failure(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)

        with patch(
            "agents.star.orchestrator.AgenteConsultorStar"
        ) as MockConsultant, patch(
            "agents.star.orchestrator.AgenteAnalisadorStar"
        ):
            MockConsultant.return_value.query.side_effect = RuntimeError(
                "Neo4j connection failed"
            )

            with pytest.raises(RuntimeError, match="Neo4j connection failed"):
                orch.run("analysis-1", params, ws_queue)

            # Error event should be in the queue
            assert not ws_queue.empty()
            event = ws_queue.get_nowait()
            assert event["analysisId"] == "analysis-1"
            assert event["architecture"] == "star"
            assert event["type"] == "error"
            assert "Neo4j connection failed" in event["payload"]

    def test_sends_error_event_on_analyzer_failure(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)

        with patch(
            "agents.star.orchestrator.AgenteConsultorStar"
        ) as MockConsultant, patch(
            "agents.star.orchestrator.AgenteAnalisadorStar"
        ) as MockAnalyzer:
            MockConsultant.return_value.query.return_value = {
                "despesas": [],
                "indicadores": [],
            }
            MockAnalyzer.return_value.analyze.side_effect = ValueError(
                "Analysis failed"
            )

            with pytest.raises(ValueError, match="Analysis failed"):
                orch.run("analysis-1", params, ws_queue)

            # Drain queue to find error event (metrics persist may have added items)
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())

            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) == 1
            assert error_events[0]["analysisId"] == "analysis-1"
            assert error_events[0]["architecture"] == "star"
            assert "Analysis failed" in error_events[0]["payload"]

    def test_reraises_exception_after_error_event(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)

        with patch(
            "agents.star.orchestrator.AgenteConsultorStar"
        ) as MockConsultant, patch(
            "agents.star.orchestrator.AgenteAnalisadorStar"
        ):
            MockConsultant.return_value.query.side_effect = RuntimeError("boom")

            with pytest.raises(RuntimeError, match="boom"):
                orch.run("analysis-1", params, ws_queue)


class TestExportsFromInit:
    """Verify __init__.py exports all three star agent classes."""

    def test_import_all_star_agents(self):
        from agents.star import (
            AgenteConsultorStar,
            AgenteAnalisadorStar,
            OrquestradorEstrela,
        )
        assert AgenteConsultorStar is not None
        assert AgenteAnalisadorStar is not None
        assert OrquestradorEstrela is not None
