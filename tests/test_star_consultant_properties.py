# Feature: multiagent-architecture-comparison, Property 1: Consulta produz estrutura válida
"""
Property-based tests for AgenteConsultorStar.query().

**Validates: Requirements 3.1, 3.2**

Verifies that for any valid period and health parameters, the consultant
agent returns data with the expected structure: each despesa contains
subfuncao (int), ano (int), and valor (numeric); each indicador contains
tipo (str), ano (int), and valor (numeric).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

from agents.star.consultant import AgenteConsultorStar

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
    """Generate a valid (date_from, date_to) pair with date_from <= date_to."""
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
    """Generate a single valid despesa record as returned by Neo4j."""
    subfuncao = draw(st.sampled_from(VALID_SUBFUNCOES))
    return {
        "subfuncao": subfuncao,
        "subfuncaoNome": SUBFUNCAO_NOMES[subfuncao],
        "ano": draw(st.integers(min_value=2000, max_value=2030)),
        "valor": draw(st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)),
    }


@st.composite
def valid_indicador_record(draw, tipos=None):
    """Generate a single valid indicador record as returned by Neo4j."""
    if tipos is None:
        tipo = draw(st.sampled_from(VALID_HEALTH_PARAMS))
    else:
        tipo = draw(st.sampled_from(tipos))
    return {
        "tipo": tipo,
        "ano": draw(st.integers(min_value=2000, max_value=2030)),
        "valor": draw(st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False)),
    }


@st.composite
def query_scenario(draw):
    """Generate a complete scenario: params + mock data for Neo4j."""
    analysis_id = draw(valid_analysis_id())
    date_from, date_to = draw(valid_period())
    health_params = draw(valid_health_params())

    despesas = draw(st.lists(valid_despesa_record(), min_size=0, max_size=10))
    indicadores = draw(
        st.lists(valid_indicador_record(tipos=health_params), min_size=0, max_size=10)
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
@given(scenario=query_scenario())
def test_p1_consulta_produz_estrutura_valida(scenario):
    """
    Property 1: Consulta produz estrutura válida.

    **Validates: Requirements 3.1, 3.2**

    For any valid period and health parameters, the AgenteConsultorStar.query()
    must return a dict with keys "despesas" and "indicadores" where:
    - Each despesa has subfuncao (int), ano (int), valor (numeric)
    - Each indicador has tipo (str), ano (int), valor (numeric)
    - The returned data matches what the Neo4j mock provided
    """
    mock_neo4j = MagicMock()
    mock_neo4j.get_despesas.return_value = scenario["despesas"]
    mock_neo4j.get_indicadores.return_value = scenario["indicadores"]

    agent = AgenteConsultorStar("consul-prop-1", mock_neo4j)
    result = agent.query(
        scenario["analysis_id"],
        scenario["date_from"],
        scenario["date_to"],
        scenario["health_params"],
    )

    # --- Structure checks ---
    assert isinstance(result, dict), "Result must be a dict"
    assert "despesas" in result, "Result must contain 'despesas' key"
    assert "indicadores" in result, "Result must contain 'indicadores' key"

    # --- Despesas validation ---
    for desp in result["despesas"]:
        assert "subfuncao" in desp, "Each despesa must have 'subfuncao'"
        assert "ano" in desp, "Each despesa must have 'ano'"
        assert "valor" in desp, "Each despesa must have 'valor'"
        assert isinstance(desp["subfuncao"], int), "subfuncao must be int"
        assert isinstance(desp["ano"], int), "ano must be int"
        assert isinstance(desp["valor"], (int, float)), "valor must be numeric"

    # --- Indicadores validation ---
    for ind in result["indicadores"]:
        assert "tipo" in ind, "Each indicador must have 'tipo'"
        assert "ano" in ind, "Each indicador must have 'ano'"
        assert "valor" in ind, "Each indicador must have 'valor'"
        assert isinstance(ind["tipo"], str), "tipo must be str"
        assert isinstance(ind["ano"], int), "ano must be int"
        assert isinstance(ind["valor"], (int, float)), "valor must be numeric"

    # --- Data fidelity: returned data matches mock ---
    assert result["despesas"] == scenario["despesas"], \
        "Returned despesas must match Neo4j data"
    assert result["indicadores"] == scenario["indicadores"], \
        "Returned indicadores must match Neo4j data"
