# Feature: multiagent-architecture-comparison, Property 6: Correlações válidas e classificação
"""
Property-based tests for AgenteAnalisadorStar correlation and classification.

**Validates: Requirements 4.1, 4.4**

Verifies that for any valid normalized dataset of despesas and indicadores,
the analyzer produces correlation values in [-1, 1] and classifications
within the allowed set {"alta", "média", "baixa"}.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from queue import Queue

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agents.star.analyzer import (
    AgenteAnalisadorStar,
    SUBFUNCAO_INDICADOR_MAP,
    _classify,
    _pearson,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SUBFUNCOES = list(SUBFUNCAO_INDICADOR_MAP.keys())
VALID_CLASSIFICATIONS = {"alta", "média", "baixa"}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_float = st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)


@st.composite
def despesa_record(draw, subfuncao, year):
    """Generate a despesa record for a given subfuncao and year."""
    return {
        "subfuncao": subfuncao,
        "ano": year,
        "valor": draw(finite_float),
    }


@st.composite
def indicador_record(draw, tipo, year):
    """Generate an indicador record for a given tipo and year."""
    return {
        "tipo": tipo,
        "ano": year,
        "valor": draw(finite_float),
    }


@st.composite
def normalized_dataset(draw):
    """Generate a dataset with overlapping years between despesas and indicadores.

    Ensures at least one subfuncao-indicador pair shares >= 2 common years
    so that a meaningful correlation can be computed.
    """
    subfuncao = draw(st.sampled_from(VALID_SUBFUNCOES))
    tipos = SUBFUNCAO_INDICADOR_MAP[subfuncao]
    tipo = draw(st.sampled_from(tipos))

    n_years = draw(st.integers(min_value=2, max_value=8))
    base_year = draw(st.integers(min_value=2000, max_value=2025))
    years = list(range(base_year, base_year + n_years))

    despesas = [draw(despesa_record(subfuncao, y)) for y in years]
    indicadores = [draw(indicador_record(tipo, y)) for y in years]

    # Optionally add extra records for other subfuncoes (noise)
    extra_despesas = draw(st.lists(
        st.builds(
            lambda sf, y, v: {"subfuncao": sf, "ano": y, "valor": v},
            st.sampled_from(VALID_SUBFUNCOES),
            st.integers(min_value=2000, max_value=2030),
            finite_float,
        ),
        max_size=5,
    ))

    return {
        "despesas": despesas + extra_despesas,
        "indicadores": indicadores,
    }


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=normalized_dataset())
def test_p6_correlacoes_validas_e_classificacao(data):
    """
    Property 6: Correlações válidas e classificação.

    **Validates: Requirements 4.1, 4.4**

    For any normalized dataset containing despesas and indicadores,
    the analyzer must produce:
    - correlation values in the interval [-1, 1]
    - classification in {"alta", "média", "baixa"}
    """
    agent = AgenteAnalisadorStar("prop6-analyzer")
    ws_queue = Queue()

    agent.update_beliefs({
        "analysis_id": "prop6-test",
        "despesas": data["despesas"],
        "indicadores": data["indicadores"],
        "ws_queue": ws_queue,
    })

    # Run only the cross-data and correlation steps
    agent._cross_data()
    agent._compute_correlations()

    correlacoes = agent.beliefs.get("correlacoes", [])

    # Must produce at least one correlation (dataset guarantees overlap)
    assert len(correlacoes) >= 1, "Expected at least one correlation from overlapping data"

    for corr in correlacoes:
        # Correlation value in [-1, 1]
        r = corr["correlacao"]
        assert -1.0 <= r <= 1.0, f"Correlation {r} outside [-1, 1]"

        # Classification is one of the allowed values
        classificacao = corr["classificacao"]
        assert classificacao in VALID_CLASSIFICATIONS, (
            f"Classification '{classificacao}' not in {VALID_CLASSIFICATIONS}"
        )

        # Classification is consistent with the correlation value
        abs_r = abs(r)
        if abs_r >= 0.7:
            assert classificacao == "alta"
        elif abs_r >= 0.4:
            assert classificacao == "média"
        else:
            assert classificacao == "baixa"

        # Structural fields present
        assert "subfuncao" in corr
        assert "tipo_indicador" in corr
        assert "n_pontos" in corr
        assert corr["n_pontos"] >= 2


@settings(max_examples=100)
@given(
    xs=st.lists(finite_float, min_size=2, max_size=20),
    ys=st.lists(finite_float, min_size=2, max_size=20),
)
def test_p6_pearson_always_in_range(xs, ys):
    """
    Property 6 (auxiliary): _pearson always returns a value in [-1, 1].

    For any two lists of finite floats, the Pearson function must return
    a value clamped to [-1, 1] or 0.0 for degenerate inputs.
    """
    min_len = min(len(xs), len(ys))
    xs = xs[:min_len]
    ys = ys[:min_len]

    r = _pearson(xs, ys)
    assert -1.0 <= r <= 1.0, f"_pearson returned {r} outside [-1, 1]"


@settings(max_examples=100)
@given(r=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
def test_p6_classify_always_valid(r):
    """
    Property 6 (auxiliary): _classify always returns a valid classification.

    For any correlation value in [-1, 1], the classification must be
    one of {"alta", "média", "baixa"}.
    """
    result = _classify(r)
    assert result in VALID_CLASSIFICATIONS, (
        f"_classify({r}) returned '{result}' not in {VALID_CLASSIFICATIONS}"
    )
