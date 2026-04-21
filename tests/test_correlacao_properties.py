# Feature: agent-specialization-remodel, Property 3: Correlation metrics are valid and consistently classified
# Feature: agent-specialization-remodel, Property 4: Insufficient data produces zero correlations
"""
Property-based tests for AgenteCorrelacao.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7**

Property 3: For any list of crossed data points with at least one pair
having >= 2 common years, all Pearson/Spearman/Kendall values are in [-1, 1],
classification is one of {"alta", "média", "baixa"}, classification matches
Spearman thresholds, and each record contains all required fields.

Property 4: For any list of crossed data points where a given
subfunção-indicador pair has fewer than 2 data points, all correlation
metrics are 0.0.
"""

from __future__ import annotations

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agents.analytical.correlacao import AgenteCorrelacao, _classify

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = {"alta", "média", "baixa"}
REQUIRED_FIELDS = {
    "subfuncao", "tipo_indicador", "pearson", "spearman",
    "kendall", "classificacao", "n_pontos",
}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

finite_positive_float = st.floats(
    min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False,
)

non_negative_float = st.floats(
    min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False,
)


@st.composite
def crossed_data_strategy(draw):
    """Generate a single CrossedDataPoint dict."""
    subfuncao = draw(st.sampled_from([301, 302, 305]))
    tipo = draw(st.sampled_from(["dengue", "covid", "vacinacao", "internacoes"]))
    return {
        "subfuncao": subfuncao,
        "subfuncao_nome": f"Subfunção {subfuncao}",
        "tipo_indicador": tipo,
        "ano": draw(st.integers(min_value=2015, max_value=2025)),
        "valor_despesa": draw(finite_positive_float),
        "valor_indicador": draw(non_negative_float),
    }


@st.composite
def crossed_data_with_sufficient_pair(draw):
    """Generate crossed data ensuring at least one pair has >= 2 data points.

    Picks a fixed subfuncao-indicador pair and generates 2-6 data points
    for it with distinct years, then optionally adds noise from other pairs.
    """
    subfuncao = draw(st.sampled_from([301, 302, 305]))
    tipo = draw(st.sampled_from(["dengue", "covid", "vacinacao", "internacoes"]))

    n_points = draw(st.integers(min_value=2, max_value=6))
    base_year = draw(st.integers(min_value=2015, max_value=2020))
    years = list(range(base_year, base_year + n_points))

    # Core pair with guaranteed >= 2 points
    core_points = []
    for y in years:
        core_points.append({
            "subfuncao": subfuncao,
            "subfuncao_nome": f"Subfunção {subfuncao}",
            "tipo_indicador": tipo,
            "ano": y,
            "valor_despesa": draw(finite_positive_float),
            "valor_indicador": draw(non_negative_float),
        })

    # Optional noise from other pairs
    noise = draw(st.lists(crossed_data_strategy(), max_size=5))

    return core_points + noise


@st.composite
def crossed_data_insufficient_pair(draw):
    """Generate crossed data where a specific pair has exactly 0 or 1 point.

    Returns (dados, target_subfuncao, target_tipo) so the test can check
    that the target pair has zero correlations.
    """
    subfuncao = draw(st.sampled_from([301, 302, 305]))
    tipo = draw(st.sampled_from(["dengue", "covid", "vacinacao", "internacoes"]))

    # 0 or 1 point for the target pair
    n_points = draw(st.integers(min_value=0, max_value=1))
    target_points = []
    for _ in range(n_points):
        target_points.append({
            "subfuncao": subfuncao,
            "subfuncao_nome": f"Subfunção {subfuncao}",
            "tipo_indicador": tipo,
            "ano": draw(st.integers(min_value=2015, max_value=2025)),
            "valor_despesa": draw(finite_positive_float),
            "valor_indicador": draw(non_negative_float),
        })

    # Add noise from OTHER pairs (different subfuncao or tipo) to ensure
    # the agent still processes something
    other_subfuncoes = [s for s in [301, 302, 305] if s != subfuncao]
    other_tipos = [t for t in ["dengue", "covid", "vacinacao", "internacoes"] if t != tipo]

    noise = draw(st.lists(
        st.builds(
            lambda sf, tp, y, vd, vi: {
                "subfuncao": sf,
                "subfuncao_nome": f"Subfunção {sf}",
                "tipo_indicador": tp,
                "ano": y,
                "valor_despesa": vd,
                "valor_indicador": vi,
            },
            st.sampled_from(other_subfuncoes) if other_subfuncoes else st.just(999),
            st.sampled_from(other_tipos) if other_tipos else st.just("outro"),
            st.integers(min_value=2015, max_value=2025),
            finite_positive_float,
            non_negative_float,
        ),
        max_size=5,
    ))

    return target_points + noise, subfuncao, tipo


# ---------------------------------------------------------------------------
# Property 3: Correlation metrics are valid and consistently classified
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p3_correlation_values_in_valid_range(dados):
    """
    Property 3: All Pearson, Spearman, and Kendall values must be in [-1, 1].

    **Validates: Requirements 5.1, 5.2, 5.3**

    For any list of crossed data points, every correlation metric produced
    by AgenteCorrelacao.compute() must lie within the interval [-1, 1].
    """
    agent = AgenteCorrelacao("prop3-corr")
    result = agent.compute(dados)

    for record in result:
        assert -1.0 <= record["pearson"] <= 1.0, (
            f"Pearson {record['pearson']} outside [-1, 1]"
        )
        assert -1.0 <= record["spearman"] <= 1.0, (
            f"Spearman {record['spearman']} outside [-1, 1]"
        )
        assert -1.0 <= record["kendall"] <= 1.0, (
            f"Kendall {record['kendall']} outside [-1, 1]"
        )


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p3_classification_is_valid_and_consistent(dados):
    """
    Property 3: Classification must be one of {"alta", "média", "baixa"}
    and must match the Spearman value according to thresholds.

    **Validates: Requirements 5.4, 5.5**

    |r| >= 0.7 → "alta", |r| >= 0.4 → "média", else "baixa"
    """
    agent = AgenteCorrelacao("prop3-corr")
    result = agent.compute(dados)

    for record in result:
        classificacao = record["classificacao"]
        assert classificacao in VALID_CLASSIFICATIONS, (
            f"Classification '{classificacao}' not in {VALID_CLASSIFICATIONS}"
        )

        # Classification must match Spearman thresholds
        abs_spearman = abs(record["spearman"])
        if abs_spearman >= 0.7:
            assert classificacao == "alta", (
                f"|spearman|={abs_spearman:.4f} should be 'alta', got '{classificacao}'"
            )
        elif abs_spearman >= 0.4:
            assert classificacao == "média", (
                f"|spearman|={abs_spearman:.4f} should be 'média', got '{classificacao}'"
            )
        else:
            assert classificacao == "baixa", (
                f"|spearman|={abs_spearman:.4f} should be 'baixa', got '{classificacao}'"
            )


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p3_required_fields_present(dados):
    """
    Property 3: Each record must contain all required fields.

    **Validates: Requirements 5.5**

    Required: subfuncao, tipo_indicador, pearson, spearman, kendall,
    classificacao, n_pontos.
    """
    agent = AgenteCorrelacao("prop3-corr")
    result = agent.compute(dados)

    assert len(result) >= 1, "Expected at least one correlation record"

    for record in result:
        missing = REQUIRED_FIELDS - set(record.keys())
        assert not missing, f"Missing required fields: {missing}"
        assert isinstance(record["subfuncao"], int)
        assert isinstance(record["tipo_indicador"], str)
        assert isinstance(record["pearson"], float)
        assert isinstance(record["spearman"], float)
        assert isinstance(record["kendall"], float)
        assert isinstance(record["classificacao"], str)
        assert isinstance(record["n_pontos"], int)
        assert record["n_pontos"] >= 1


# ---------------------------------------------------------------------------
# Property 4: Insufficient data produces zero correlations
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=crossed_data_insufficient_pair())
def test_p4_insufficient_data_produces_zero_correlations(data):
    """
    Property 4: Pairs with < 2 data points must have all metrics = 0.0.

    **Validates: Requirements 5.7**

    For any list of crossed data points where a given subfunção-indicador
    pair has fewer than 2 data points, pearson = 0.0, spearman = 0.0,
    kendall = 0.0.
    """
    dados, target_subfuncao, target_tipo = data

    agent = AgenteCorrelacao("prop4-corr")
    result = agent.compute(dados)

    # Find the target pair in results (may not exist if 0 points)
    target_records = [
        r for r in result
        if r["subfuncao"] == target_subfuncao and r["tipo_indicador"] == target_tipo
    ]

    for record in target_records:
        assert record["n_pontos"] < 2, (
            f"Expected < 2 points for target pair, got {record['n_pontos']}"
        )
        assert record["pearson"] == 0.0, (
            f"Expected pearson=0.0 for insufficient data, got {record['pearson']}"
        )
        assert record["spearman"] == 0.0, (
            f"Expected spearman=0.0 for insufficient data, got {record['spearman']}"
        )
        assert record["kendall"] == 0.0, (
            f"Expected kendall=0.0 for insufficient data, got {record['kendall']}"
        )


@settings(max_examples=100)
@given(r=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False))
def test_p3_classify_always_returns_valid_classification(r):
    """
    Property 3 (auxiliary): _classify always returns a valid classification.

    **Validates: Requirements 5.4**

    For any correlation value in [-1, 1], the classification must be
    one of {"alta", "média", "baixa"} and consistent with thresholds.
    """
    result = _classify(r)
    assert result in VALID_CLASSIFICATIONS, (
        f"_classify({r}) returned '{result}' not in {VALID_CLASSIFICATIONS}"
    )

    abs_r = abs(r)
    if abs_r >= 0.7:
        assert result == "alta"
    elif abs_r >= 0.4:
        assert result == "média"
    else:
        assert result == "baixa"
