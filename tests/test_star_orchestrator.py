"""
Testes unitários para OrquestradorEstrela (reescrito com 8 agentes especializados).

Valida: Requisitos 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 11.1, 11.2
"""

from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from agents.base import AgenteBDI
from agents.star.orchestrator import OrquestradorEstrela


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


def _patch_all_agents():
    """Context manager that patches all 8 specialized agent classes."""
    return (
        patch("agents.star.orchestrator.AgenteVigilanciaEpidemiologica"),
        patch("agents.star.orchestrator.AgenteSaudeHospitalar"),
        patch("agents.star.orchestrator.AgenteAtencaoPrimaria"),
        patch("agents.star.orchestrator.AgenteMortalidade"),
        patch("agents.star.orchestrator.AgenteContextoOrcamentario"),
        patch("agents.star.orchestrator.AgenteCorrelacao"),
        patch("agents.star.orchestrator.AgenteAnomalias"),
        patch("agents.star.orchestrator.AgenteSintetizador"),
    )


def _setup_domain_mocks(mock_vig, mock_hosp, mock_prim, mock_mort):
    """Configure domain agent mocks to return valid data."""
    mock_vig.return_value.query.return_value = {
        "despesas": [{"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 500.0}],
        "indicadores": [{"tipo": "dengue", "ano": 2020, "valor": 150.0}],
    }
    mock_hosp.return_value.query.return_value = {
        "despesas": [{"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 800.0}],
        "indicadores": [{"tipo": "internacoes", "ano": 2020, "valor": 200.0}],
    }
    mock_prim.return_value.query.return_value = {
        "despesas": [{"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 1000.0}],
        "indicadores": [{"tipo": "vacinacao", "ano": 2020, "valor": 80.0}],
    }
    mock_mort.return_value.query.return_value = {
        "despesas": [],
        "indicadores": [{"tipo": "mortalidade", "ano": 2020, "valor": 50.0}],
    }


def _setup_analytical_mocks(mock_ctx, mock_corr, mock_anom, mock_sint):
    """Configure analytical agent mocks to return valid data."""
    mock_ctx.return_value.analyze_trends.return_value = {
        301: {"subfuncao": 301, "tendencia": "crescimento", "variacao_media_percentual": 5.0, "anos_analisados": [2020]},
    }
    mock_corr.return_value.compute.return_value = [
        {"subfuncao": 301, "tipo_indicador": "vacinacao", "pearson": 0.8, "spearman": 0.75, "kendall": 0.7, "classificacao": "alta", "n_pontos": 3},
    ]
    mock_anom.return_value.detect.return_value = []
    mock_sint.return_value.synthesize.return_value = "Análise completa."


class TestOrquestradorInit:
    """Req 9.1: topologia estrela com OrquestradorEstrela central."""

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
        assert "executar_pipeline_dominio" in goals
        assert "executar_pipeline_analitico" in goals
        assert "persistir_metricas" in goals

    def test_deliberate_without_analysis_id(self, mock_neo4j):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        desires = orch.deliberate()
        assert desires == []


class TestRunPipeline:
    """Req 9.2, 9.3, 9.4, 9.5, 9.6: full pipeline with 8 agents."""

    def test_run_delegates_to_all_agents(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        patches = _patch_all_agents()

        with patches[0] as MockVig, patches[1] as MockHosp, \
             patches[2] as MockPrim, patches[3] as MockMort, \
             patches[4] as MockCtx, patches[5] as MockCorr, \
             patches[6] as MockAnom, patches[7] as MockSint:

            _setup_domain_mocks(MockVig, MockHosp, MockPrim, MockMort)
            _setup_analytical_mocks(MockCtx, MockCorr, MockAnom, MockSint)

            result = orch.run("analysis-1", params, ws_queue)

            # All 4 domain agents were called
            MockVig.return_value.query.assert_called_once()
            MockHosp.return_value.query.assert_called_once()
            MockPrim.return_value.query.assert_called_once()
            MockMort.return_value.query.assert_called_once()

            # Analytical agents were called
            MockCtx.return_value.analyze_trends.assert_called_once()
            MockCorr.return_value.compute.assert_called_once()
            MockAnom.return_value.detect.assert_called_once()
            MockSint.return_value.synthesize.assert_called_once()

            # Result contains all expected keys
            assert "correlacoes" in result
            assert "anomalias" in result
            assert "texto_analise" in result
            assert "despesas" in result
            assert "indicadores" in result
            assert "contexto_orcamentario" in result
            assert "message_count" in result


class TestMetricsPersistence:
    """Req 9.7: registrar métricas de tempo de execução por agente."""

    def test_persists_metrics_for_all_agents(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        patches = _patch_all_agents()

        with patches[0] as MockVig, patches[1] as MockHosp, \
             patches[2] as MockPrim, patches[3] as MockMort, \
             patches[4] as MockCtx, patches[5] as MockCorr, \
             patches[6] as MockAnom, patches[7] as MockSint:

            _setup_domain_mocks(MockVig, MockHosp, MockPrim, MockMort)
            _setup_analytical_mocks(MockCtx, MockCorr, MockAnom, MockSint)

            orch.run("analysis-1", params, ws_queue)

            # save_metrica should be called 8 times: once per agent
            assert mock_neo4j.save_metrica.call_count == 8

            # All calls should use architecture "star"
            for call in mock_neo4j.save_metrica.call_args_list:
                metrica = call[0][0]
                analysis = call[0][1]
                assert analysis == "analysis-1"
                assert metrica["architecture"] == "star"
                assert "executionTimeMs" in metrica
                assert "cpuPercent" in metrica
                assert "memoryMb" in metrica


class TestErrorHandling:
    """Req 9.8: graceful degradation when agents fail."""

    def test_continues_when_domain_agent_fails(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        patches = _patch_all_agents()

        with patches[0] as MockVig, patches[1] as MockHosp, \
             patches[2] as MockPrim, patches[3] as MockMort, \
             patches[4] as MockCtx, patches[5] as MockCorr, \
             patches[6] as MockAnom, patches[7] as MockSint:

            # Vigilancia fails
            MockVig.return_value.query.side_effect = RuntimeError("Neo4j down")
            # Others succeed
            MockHosp.return_value.query.return_value = {
                "despesas": [{"subfuncao": 302, "ano": 2020, "valor": 800.0}],
                "indicadores": [],
            }
            MockPrim.return_value.query.return_value = {"despesas": [], "indicadores": []}
            MockMort.return_value.query.return_value = {"despesas": [], "indicadores": []}
            _setup_analytical_mocks(MockCtx, MockCorr, MockAnom, MockSint)

            # Should NOT raise — graceful degradation
            result = orch.run("analysis-1", params, ws_queue)

            # Error event should be in the queue
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) >= 1
            assert error_events[0]["architecture"] == "star"

            # Pipeline still completed
            assert "correlacoes" in result
            assert "texto_analise" in result

    def test_continues_when_analytical_agent_fails(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        patches = _patch_all_agents()

        with patches[0] as MockVig, patches[1] as MockHosp, \
             patches[2] as MockPrim, patches[3] as MockMort, \
             patches[4] as MockCtx, patches[5] as MockCorr, \
             patches[6] as MockAnom, patches[7] as MockSint:

            _setup_domain_mocks(MockVig, MockHosp, MockPrim, MockMort)
            MockCtx.return_value.analyze_trends.return_value = {}
            MockCorr.return_value.compute.side_effect = ValueError("Computation error")
            MockAnom.return_value.detect.return_value = []
            MockSint.return_value.synthesize.return_value = "Partial analysis."

            result = orch.run("analysis-1", params, ws_queue)

            # Error event for correlacao failure
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) >= 1

            # Correlacoes should be empty due to failure
            assert result["correlacoes"] == []
            # But text analysis still ran
            assert result["texto_analise"] == "Partial analysis."


class TestMessageCount:
    """Req 11.1, 11.2: message counter integration."""

    def test_message_count_in_result(
        self, mock_neo4j, params, ws_queue
    ):
        orch = OrquestradorEstrela("orch-1", mock_neo4j)
        patches = _patch_all_agents()

        with patches[0] as MockVig, patches[1] as MockHosp, \
             patches[2] as MockPrim, patches[3] as MockMort, \
             patches[4] as MockCtx, patches[5] as MockCorr, \
             patches[6] as MockAnom, patches[7] as MockSint:

            _setup_domain_mocks(MockVig, MockHosp, MockPrim, MockMort)
            _setup_analytical_mocks(MockCtx, MockCorr, MockAnom, MockSint)

            result = orch.run("analysis-1", params, ws_queue)

            # message_count should be present and equal to 2 * 8 = 16
            assert "message_count" in result
            assert result["message_count"] == 16  # 8 agents × 2 messages each

            # metric event should be in the queue
            events = []
            while not ws_queue.empty():
                events.append(ws_queue.get_nowait())
            metric_events = [e for e in events if e.get("type") == "metric"]
            assert len(metric_events) >= 1


class TestExportsFromInit:
    """Verify __init__.py exports only OrquestradorEstrela."""

    def test_import_orchestrator(self):
        from agents.star import OrquestradorEstrela as OE
        assert OE is not None

    def test_all_exports(self):
        from agents.star import __all__
        assert "OrquestradorEstrela" in __all__
        # Old classes should NOT be exported
        assert "AgenteConsultorStar" not in __all__
        assert "AgenteAnalisadorStar" not in __all__
