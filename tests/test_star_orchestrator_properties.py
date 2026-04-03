# Feature: multiagent-architecture-comparison, Property 7: Comunicação estrela via orquestrador
"""
Property-based tests for star architecture communication mediation.

**Validates: Requirement 5.2**

Verifies that no peripheral agent calls another directly — all interaction
flows through the OrquestradorEstrela. Specifically:
1. The consultant never references or calls the analyzer.
2. The analyzer never references or calls the consultant.
3. All data flows through the orchestrator: consultant returns data to the
   orchestrator, which then passes it to the analyzer.
4. The orchestrator is the sole intermediary.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import inspect
from queue import Queue
from unittest.mock import MagicMock, patch, call

from hypothesis import given, settings
from hypothesis import strategies as st

from agents.star.orchestrator import OrquestradorEstrela
from agents.star.consultant import AgenteConsultorStar
from agents.star.analyzer import AgenteAnalisadorStar


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SUBFUNCOES = [301, 302, 303, 305]
SUBFUNCAO_NOMES = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}
VALID_HEALTH_PARAMS = ["dengue", "covid", "vacinacao", "mortalidade", "internacoes"]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def valid_analysis_id(draw):
    """Generate a UUID-like analysis ID string."""
    return draw(st.uuids().map(str))


@st.composite
def valid_period(draw):
    """Generate a valid (date_from, date_to) pair."""
    date_from = draw(st.integers(min_value=2000, max_value=2030))
    date_to = draw(st.integers(min_value=date_from, max_value=2030))
    return date_from, date_to


@st.composite
def valid_health_params(draw):
    """Generate a non-empty subset of valid health parameters."""
    return draw(
        st.lists(
            st.sampled_from(VALID_HEALTH_PARAMS),
            min_size=1,
            max_size=len(VALID_HEALTH_PARAMS),
            unique=True,
        )
    )


@st.composite
def valid_despesa_record(draw):
    """Generate a single valid despesa record."""
    subfuncao = draw(st.sampled_from(VALID_SUBFUNCOES))
    return {
        "subfuncao": subfuncao,
        "subfuncaoNome": SUBFUNCAO_NOMES[subfuncao],
        "ano": draw(st.integers(min_value=2000, max_value=2030)),
        "valor": draw(st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)),
    }


@st.composite
def valid_indicador_record(draw, tipos=None):
    """Generate a single valid indicador record."""
    if tipos is None:
        tipo = draw(st.sampled_from(VALID_HEALTH_PARAMS))
    else:
        tipo = draw(st.sampled_from(tipos))
    return {
        "tipo": tipo,
        "ano": draw(st.integers(min_value=2000, max_value=2030)),
        "valor": draw(st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)),
    }


@st.composite
def orchestrator_scenario(draw):
    """Generate a complete orchestrator run scenario."""
    analysis_id = draw(valid_analysis_id())
    date_from, date_to = draw(valid_period())
    health_params = draw(valid_health_params())

    despesas = draw(st.lists(valid_despesa_record(), min_size=1, max_size=8))
    indicadores = draw(
        st.lists(valid_indicador_record(tipos=health_params), min_size=1, max_size=8)
    )

    return {
        "analysis_id": analysis_id,
        "date_from": date_from,
        "date_to": date_to,
        "health_params": health_params,
        "despesas": despesas,
        "indicadores": indicadores,
    }


# ---------------------------------------------------------------------------
# Property test
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(scenario=orchestrator_scenario())
def test_p7_comunicacao_estrela_via_orquestrador(scenario):
    """
    Property 7: Comunicação estrela via orquestrador.

    **Validates: Requirement 5.2**

    For any valid analysis parameters, the OrquestradorEstrela must be the
    sole intermediary between peripheral agents:
    1. Consultant source code does not reference the analyzer class.
    2. Analyzer source code does not reference the consultant class.
    3. The orchestrator calls consultant.query() first, then passes the
       exact returned data to analyzer.analyze().
    4. No direct calls exist between peripheral agents.
    """
    # --- Static verification: peripheral agents don't reference each other ---
    consultant_source = inspect.getsource(AgenteConsultorStar)
    assert "AgenteAnalisadorStar" not in consultant_source, (
        "Consultant must not reference AgenteAnalisadorStar"
    )
    assert "analyzer" not in consultant_source.lower().replace("analisador", "").replace("análise", ""), (
        "Consultant must not reference analyzer"
    )

    analyzer_source = inspect.getsource(AgenteAnalisadorStar)
    assert "AgenteConsultorStar" not in analyzer_source, (
        "Analyzer must not reference AgenteConsultorStar"
    )
    assert "consultant" not in analyzer_source.lower().replace("consulta", ""), (
        "Analyzer must not reference consultant"
    )

    # --- Dynamic verification: orchestrator mediates all data flow ---
    mock_neo4j = MagicMock()
    ws_queue = Queue()

    consultant_data = {
        "despesas": scenario["despesas"],
        "indicadores": scenario["indicadores"],
    }

    with patch(
        "agents.star.orchestrator.AgenteConsultorStar"
    ) as MockConsultant, patch(
        "agents.star.orchestrator.AgenteAnalisadorStar"
    ) as MockAnalyzer:
        mock_consultant_instance = MockConsultant.return_value
        mock_consultant_instance.query.return_value = consultant_data

        mock_analyzer_instance = MockAnalyzer.return_value
        mock_analyzer_instance.analyze.return_value = {
            "correlacoes": [],
            "anomalias": [],
            "texto_analise": "Análise completa.",
        }

        orch = OrquestradorEstrela("orch-prop7", mock_neo4j)
        params = {
            "date_from": scenario["date_from"],
            "date_to": scenario["date_to"],
            "health_params": scenario["health_params"],
        }

        orch.run(scenario["analysis_id"], params, ws_queue)

        # 1. Orchestrator called consultant.query() exactly once
        mock_consultant_instance.query.assert_called_once()

        # 2. Orchestrator called analyzer.analyze() exactly once
        mock_analyzer_instance.analyze.assert_called_once()

        # 3. Data mediation: the exact data returned by consultant was
        #    passed to analyzer by the orchestrator
        analyzer_call_args = mock_analyzer_instance.analyze.call_args
        actual_data_passed = analyzer_call_args[0][1]
        assert actual_data_passed is consultant_data, (
            "Orchestrator must pass the exact consultant data to analyzer — "
            "data must flow through the orchestrator, not directly"
        )

        # 4. The ws_queue passed to analyzer is the same one the orchestrator received
        actual_queue_passed = analyzer_call_args[0][2]
        assert actual_queue_passed is ws_queue, (
            "Orchestrator must pass the original ws_queue to analyzer"
        )

        # 5. Peripheral agents never called each other:
        #    - consultant instance has no calls to analyzer methods
        #    - analyzer instance has no calls to consultant methods
        consultant_calls = [
            str(c) for c in mock_consultant_instance.method_calls
        ]
        for c in consultant_calls:
            assert "analyze" not in c, (
                "Consultant must not call analyze — only orchestrator mediates"
            )

        analyzer_calls = [
            str(c) for c in mock_analyzer_instance.method_calls
        ]
        for c in analyzer_calls:
            assert "query" not in c, (
                "Analyzer must not call query — only orchestrator mediates"
            )
