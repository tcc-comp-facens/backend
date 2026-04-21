# Feature: agent-specialization-remodel, Property 8: Budget trend computation correctness
"""
Property-based tests for AgenteContextoOrcamentario.

**Validates: Requirements 8.1, 8.2, 8.3, 8.5**

Property 8: For any list of despesas spanning multiple years and subfunções:
1. Year-over-year percentage variation is computed correctly as
   ((valor_n - valor_n-1) / valor_n-1) × 100
2. Trends are classified correctly:
   - "crescimento" when consecutive positive variations ≥ 2 years
   - "corte" when consecutive negative variations ≥ 2 years
   - "estagnacao" when |variation| < 5%
3. Subfunções with fewer than 2 years of data return "insuficiente"
4. Each trend record contains required fields
   (subfuncao, tendencia, variacao_media_percentual, anos_analisados)
"""

from __future__ import annotations

import math

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from agents.context.contexto_orcamentario import (
    AgenteContextoOrcamentario,
    _compute_yoy_variation,
    _classify_trend,
    STAGNATION_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_TENDENCIAS = {"crescimento", "corte", "estagnacao", "insuficiente"}
REQUIRED_FIELDS = {"subfuncao", "tendencia", "variacao_media_percentual", "anos_analisados"}

SUBFUNCOES = [301, 302, 303, 305]
SUBFUNCAO_NOMES = [
    "Atenção Básica",
    "Assistência Hospitalar",
    "Suporte Profilático",
    "Vigilância Epidemiológica",
]

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def despesa_strategy(draw):
    """Generate a single DespesaRecord dict."""
    return {
        "subfuncao": draw(st.sampled_from(SUBFUNCOES)),
        "subfuncaoNome": draw(st.sampled_from(SUBFUNCAO_NOMES)),
        "ano": draw(st.integers(min_value=2015, max_value=2025)),
        "valor": draw(
            st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)
        ),
    }


@st.composite
def despesas_with_multi_year_subfuncao(draw):
    """Generate despesas ensuring at least one subfunção has ≥ 2 distinct years.

    Picks a fixed subfunção and generates 2-6 data points with distinct years,
    then optionally adds noise from other subfunções.
    """
    subfuncao = draw(st.sampled_from(SUBFUNCOES))
    nome = SUBFUNCAO_NOMES[SUBFUNCOES.index(subfuncao)]

    n_points = draw(st.integers(min_value=2, max_value=6))
    base_year = draw(st.integers(min_value=2015, max_value=2020))
    years = list(range(base_year, base_year + n_points))

    core_points = []
    for y in years:
        core_points.append({
            "subfuncao": subfuncao,
            "subfuncaoNome": nome,
            "ano": y,
            "valor": draw(
                st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)
            ),
        })

    # Optional noise from other subfunções
    noise = draw(st.lists(despesa_strategy(), max_size=5))

    return core_points + noise, subfuncao


@st.composite
def despesas_single_year_subfuncao(draw):
    """Generate despesas where a target subfunção has exactly 1 year of data.

    Returns (despesas, target_subfuncao).
    """
    subfuncao = draw(st.sampled_from(SUBFUNCOES))
    nome = SUBFUNCAO_NOMES[SUBFUNCOES.index(subfuncao)]

    target_point = {
        "subfuncao": subfuncao,
        "subfuncaoNome": nome,
        "ano": draw(st.integers(min_value=2015, max_value=2025)),
        "valor": draw(
            st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False)
        ),
    }

    # Add noise from OTHER subfunções only
    other_subfuncoes = [s for s in SUBFUNCOES if s != subfuncao]
    noise = draw(st.lists(
        st.builds(
            lambda sf, y, v: {
                "subfuncao": sf,
                "subfuncaoNome": SUBFUNCAO_NOMES[SUBFUNCOES.index(sf)],
                "ano": y,
                "valor": v,
            },
            st.sampled_from(other_subfuncoes) if other_subfuncoes else st.just(999),
            st.integers(min_value=2015, max_value=2025),
            st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False),
        ),
        max_size=5,
    ))

    return [target_point] + noise, subfuncao


# ---------------------------------------------------------------------------
# Property 8.1: Year-over-year variation formula correctness
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    valor_current=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False),
    valor_previous=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False),
)
def test_p8_yoy_variation_formula_correctness(valor_current, valor_previous):
    """
    Property 8: YoY variation is computed as ((valor_n - valor_n-1) / valor_n-1) × 100.

    **Validates: Requirements 8.1**

    For any two positive float values, _compute_yoy_variation must return
    the correct percentage variation.
    """
    result = _compute_yoy_variation(valor_current, valor_previous)
    expected = ((valor_current - valor_previous) / valor_previous) * 100.0

    assert math.isfinite(result), f"Expected finite result, got {result}"
    assert abs(result - expected) < 1e-6, (
        f"YoY variation mismatch: got {result}, expected {expected}"
    )


@settings(max_examples=100)
@given(data=despesas_with_multi_year_subfuncao())
def test_p8_yoy_variation_applied_correctly_in_trends(data):
    """
    Property 8: analyze_trends computes YoY variations correctly for each subfunção.

    **Validates: Requirements 8.1**

    For any subfunção with ≥ 2 years, the agent must produce a trend record
    whose variacao_media_percentual is consistent with the YoY formula.
    """
    despesas, target_subfuncao = data

    agent = AgenteContextoOrcamentario("prop8-ctx")
    result = agent.analyze_trends(despesas)

    assert target_subfuncao in result, (
        f"Expected subfuncao {target_subfuncao} in results"
    )

    record = result[target_subfuncao]

    # Aggregate values by year for the target subfunção
    year_values: dict[int, float] = {}
    for d in despesas:
        if d["subfuncao"] == target_subfuncao:
            year_values[d["ano"]] = year_values.get(d["ano"], 0.0) + d["valor"]

    sorted_years = sorted(year_values.keys())

    if len(sorted_years) >= 2:
        # Manually compute expected variations
        expected_variations = []
        for i in range(1, len(sorted_years)):
            prev_val = year_values[sorted_years[i - 1]]
            curr_val = year_values[sorted_years[i]]
            var = _compute_yoy_variation(curr_val, prev_val)
            expected_variations.append(var)

        finite_vars = [v for v in expected_variations if math.isfinite(v)]
        if finite_vars:
            expected_avg = sum(finite_vars) / len(finite_vars)
            actual_avg = record["variacao_media_percentual"]
            if math.isfinite(actual_avg):
                assert abs(actual_avg - round(expected_avg, 2)) < 0.02, (
                    f"Average variation mismatch: got {actual_avg}, "
                    f"expected ~{round(expected_avg, 2)}"
                )


# ---------------------------------------------------------------------------
# Property 8.2: Trend classification correctness
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=despesas_with_multi_year_subfuncao())
def test_p8_trend_classification_is_valid(data):
    """
    Property 8: Trend classification must be one of the valid tendências.

    **Validates: Requirements 8.2**

    For any set of despesas, every trend record must have a tendencia
    in {"crescimento", "corte", "estagnacao", "insuficiente"}.
    """
    despesas, _ = data

    agent = AgenteContextoOrcamentario("prop8-cls")
    result = agent.analyze_trends(despesas)

    for sf, record in result.items():
        assert record["tendencia"] in VALID_TENDENCIAS, (
            f"Subfuncao {sf}: tendencia '{record['tendencia']}' "
            f"not in {VALID_TENDENCIAS}"
        )


@settings(max_examples=100)
@given(
    variations=st.lists(
        st.floats(
            min_value=-STAGNATION_THRESHOLD + 0.01,
            max_value=STAGNATION_THRESHOLD - 0.01,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=10,
    )
)
def test_p8_classify_trend_stagnation(variations):
    """
    Property 8: When all |variation| < 5%, trend must be "estagnacao".

    **Validates: Requirements 8.2**
    """
    result = _classify_trend(variations)
    assert result == "estagnacao", (
        f"All variations within ±{STAGNATION_THRESHOLD}% but got '{result}' "
        f"instead of 'estagnacao'. Variations: {variations}"
    )


@settings(max_examples=100)
@given(
    n_positive=st.integers(min_value=2, max_value=8),
)
def test_p8_classify_trend_crescimento(n_positive):
    """
    Property 8: Consecutive positive variations ≥ 2 → "crescimento".

    **Validates: Requirements 8.2**
    """
    # Build a list of strictly positive variations (≥ 2 consecutive)
    variations = [10.0 + i for i in range(n_positive)]

    result = _classify_trend(variations)
    assert result == "crescimento", (
        f"{n_positive} consecutive positive variations but got '{result}' "
        f"instead of 'crescimento'. Variations: {variations}"
    )


@settings(max_examples=100)
@given(
    n_negative=st.integers(min_value=2, max_value=8),
)
def test_p8_classify_trend_corte(n_negative):
    """
    Property 8: Consecutive negative variations ≥ 2 → "corte".

    **Validates: Requirements 8.2**
    """
    # Build a list of strictly negative variations (≥ 2 consecutive)
    variations = [-(10.0 + i) for i in range(n_negative)]

    result = _classify_trend(variations)
    assert result == "corte", (
        f"{n_negative} consecutive negative variations but got '{result}' "
        f"instead of 'corte'. Variations: {variations}"
    )


# ---------------------------------------------------------------------------
# Property 8.3: Insufficient data returns "insuficiente"
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=despesas_single_year_subfuncao())
def test_p8_insufficient_data_returns_insuficiente(data):
    """
    Property 8: Subfunções with < 2 years of data return "insuficiente".

    **Validates: Requirements 8.5**

    For any subfunção with exactly 1 year of data, the trend must be
    classified as "insuficiente".
    """
    despesas, target_subfuncao = data

    agent = AgenteContextoOrcamentario("prop8-insuf")
    result = agent.analyze_trends(despesas)

    if target_subfuncao in result:
        record = result[target_subfuncao]
        assert record["tendencia"] == "insuficiente", (
            f"Subfuncao {target_subfuncao} has 1 year but tendencia is "
            f"'{record['tendencia']}' instead of 'insuficiente'"
        )
        assert record["variacao_media_percentual"] == 0.0, (
            f"Expected variacao_media_percentual=0.0 for insufficient data, "
            f"got {record['variacao_media_percentual']}"
        )


@settings(max_examples=100)
@given(
    subfuncao=st.sampled_from(SUBFUNCOES),
)
def test_p8_empty_input_returns_empty(subfuncao):
    """
    Property 8: Empty despesas list returns empty result dict.

    **Validates: Requirements 8.5**
    """
    agent = AgenteContextoOrcamentario("prop8-empty")
    result = agent.analyze_trends([])

    assert result == {}, f"Expected empty dict for empty input, got {result}"


# ---------------------------------------------------------------------------
# Property 8.5: Required fields present in each trend record
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(data=despesas_with_multi_year_subfuncao())
def test_p8_required_fields_present(data):
    """
    Property 8: Each trend record contains all required fields.

    **Validates: Requirements 8.3**

    Required: subfuncao, tendencia, variacao_media_percentual, anos_analisados.
    """
    despesas, _ = data

    agent = AgenteContextoOrcamentario("prop8-fields")
    result = agent.analyze_trends(despesas)

    assert len(result) >= 1, "Expected at least one trend record"

    for sf, record in result.items():
        missing = REQUIRED_FIELDS - set(record.keys())
        assert not missing, f"Subfuncao {sf}: missing required fields: {missing}"

        assert isinstance(record["subfuncao"], int), (
            f"subfuncao should be int, got {type(record['subfuncao'])}"
        )
        assert isinstance(record["tendencia"], str), (
            f"tendencia should be str, got {type(record['tendencia'])}"
        )
        assert isinstance(record["variacao_media_percentual"], (int, float)), (
            f"variacao_media_percentual should be numeric, "
            f"got {type(record['variacao_media_percentual'])}"
        )
        assert isinstance(record["anos_analisados"], list), (
            f"anos_analisados should be list, got {type(record['anos_analisados'])}"
        )
        # anos_analisados should contain integers
        for ano in record["anos_analisados"]:
            assert isinstance(ano, int), (
                f"Each year in anos_analisados should be int, got {type(ano)}"
            )
        # anos_analisados should be sorted
        assert record["anos_analisados"] == sorted(record["anos_analisados"]), (
            f"anos_analisados should be sorted, got {record['anos_analisados']}"
        )
