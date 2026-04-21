# Feature: agent-specialization-remodel, Property 5: Anomaly detection correctness
"""
Property-based tests for AgenteAnomalias.

**Validates: Requirements 6.1, 6.2, 6.3, 6.5**

Property 5: For any list of crossed data points:
- Every "alto_gasto_baixo_resultado" anomaly must have despesa > median AND
  indicador < median for its pair.
- Every "baixo_gasto_alto_resultado" anomaly must have despesa < median AND
  indicador > median for its pair.
- Pairs with fewer than 2 data points must produce no anomalies.
- Every anomaly record must contain required fields (subfuncao, tipo_indicador,
  ano, tipo_anomalia, descricao).
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from hypothesis import given, settings
from hypothesis import strategies as st

from agents.analytical.anomalias import AgenteAnomalias, _median

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ANOMALY_TYPES = {"alto_gasto_baixo_resultado", "baixo_gasto_alto_resultado"}
REQUIRED_FIELDS = {"subfuncao", "tipo_indicador", "ano", "tipo_anomalia", "descricao"}

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
def crossed_data_insufficient_only(draw):
    """Generate crossed data where EVERY pair has fewer than 2 data points.

    Each pair gets exactly 0 or 1 data point, ensuring no pair can produce
    anomalies.
    """
    # Generate 1-4 data points, each from a DIFFERENT (subfuncao, tipo) pair
    n_pairs = draw(st.integers(min_value=1, max_value=4))
    all_pairs = [
        (sf, tp)
        for sf in [301, 302, 305]
        for tp in ["dengue", "covid", "vacinacao", "internacoes"]
    ]
    chosen_pairs = draw(
        st.lists(
            st.sampled_from(all_pairs),
            min_size=n_pairs,
            max_size=n_pairs,
            unique=True,
        )
    )

    points = []
    for sf, tp in chosen_pairs:
        points.append({
            "subfuncao": sf,
            "subfuncao_nome": f"Subfunção {sf}",
            "tipo_indicador": tp,
            "ano": draw(st.integers(min_value=2015, max_value=2025)),
            "valor_despesa": draw(finite_positive_float),
            "valor_indicador": draw(non_negative_float),
        })

    return points


# ---------------------------------------------------------------------------
# Helpers — independent median computation for verification
# ---------------------------------------------------------------------------

def _compute_pair_medians(dados: list[dict]) -> dict[tuple[int, str], tuple[float, float]]:
    """Independently compute medians per (subfuncao, tipo_indicador) pair.

    Returns dict mapping (subfuncao, tipo) -> (median_despesa, median_indicador).
    Only includes pairs with >= 2 data points.
    """
    pairs: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for item in dados:
        key = (item["subfuncao"], item["tipo_indicador"])
        pairs[key].append(item)

    result = {}
    for key, items in pairs.items():
        if len(items) < 2:
            continue
        despesas = sorted([it["valor_despesa"] for it in items])
        indicadores = sorted([it["valor_indicador"] for it in items])
        result[key] = (_independent_median(despesas), _independent_median(indicadores))
    return result


def _independent_median(values: list[float]) -> float:
    """Compute median independently (same algorithm as statistics.median)."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return sorted_vals[mid]


# ---------------------------------------------------------------------------
# Property 5a: Anomaly median conditions are correct
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p5_alto_gasto_baixo_resultado_satisfies_median_condition(dados):
    """
    Property 5: Every "alto_gasto_baixo_resultado" anomaly must have
    despesa > median AND indicador < median for its pair.

    **Validates: Requirements 6.1**

    For any list of crossed data points, we independently compute the
    median for each pair and verify the anomaly conditions hold.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect(dados)

    medians = _compute_pair_medians(dados)

    for anom in anomalias:
        if anom["tipo_anomalia"] != "alto_gasto_baixo_resultado":
            continue

        key = (anom["subfuncao"], anom["tipo_indicador"])
        assert key in medians, (
            f"Anomaly for pair {key} but pair has < 2 points or doesn't exist"
        )

        med_desp, med_ind = medians[key]

        # Find original data points for this anomaly's (subfuncao, tipo, ano).
        # Multiple items may share the same year; the agent flags each item
        # individually, so at least one must satisfy the condition.
        matching = [
            d for d in dados
            if d["subfuncao"] == anom["subfuncao"]
            and d["tipo_indicador"] == anom["tipo_indicador"]
            and d["ano"] == anom["ano"]
        ]
        assert len(matching) >= 1, (
            f"No data point found for anomaly at year {anom['ano']}"
        )

        satisfies = any(
            dp["valor_despesa"] > med_desp and dp["valor_indicador"] < med_ind
            for dp in matching
        )
        assert satisfies, (
            f"alto_gasto_baixo_resultado at year {anom['ano']}: no data point "
            f"has despesa > {med_desp} AND indicador < {med_ind}. "
            f"Points: {[(dp['valor_despesa'], dp['valor_indicador']) for dp in matching]}"
        )


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p5_baixo_gasto_alto_resultado_satisfies_median_condition(dados):
    """
    Property 5: Every "baixo_gasto_alto_resultado" anomaly must have
    despesa < median AND indicador > median for its pair.

    **Validates: Requirements 6.2**

    For any list of crossed data points, we independently compute the
    median for each pair and verify the anomaly conditions hold.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect(dados)

    medians = _compute_pair_medians(dados)

    for anom in anomalias:
        if anom["tipo_anomalia"] != "baixo_gasto_alto_resultado":
            continue

        key = (anom["subfuncao"], anom["tipo_indicador"])
        assert key in medians, (
            f"Anomaly for pair {key} but pair has < 2 points or doesn't exist"
        )

        med_desp, med_ind = medians[key]

        # Find original data points for this anomaly's (subfuncao, tipo, ano).
        # Multiple items may share the same year; the agent flags each item
        # individually, so at least one must satisfy the condition.
        matching = [
            d for d in dados
            if d["subfuncao"] == anom["subfuncao"]
            and d["tipo_indicador"] == anom["tipo_indicador"]
            and d["ano"] == anom["ano"]
        ]
        assert len(matching) >= 1, (
            f"No data point found for anomaly at year {anom['ano']}"
        )

        satisfies = any(
            dp["valor_despesa"] < med_desp and dp["valor_indicador"] > med_ind
            for dp in matching
        )
        assert satisfies, (
            f"baixo_gasto_alto_resultado at year {anom['ano']}: no data point "
            f"has despesa < {med_desp} AND indicador > {med_ind}. "
            f"Points: {[(dp['valor_despesa'], dp['valor_indicador']) for dp in matching]}"
        )


# ---------------------------------------------------------------------------
# Property 5b: Pairs with < 2 data points produce no anomalies
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dados=crossed_data_insufficient_only())
def test_p5_insufficient_data_produces_no_anomalies(dados):
    """
    Property 5: Pairs with fewer than 2 data points must produce no anomalies.

    **Validates: Requirements 6.5**

    When every (subfuncao, tipo_indicador) pair has fewer than 2 data points,
    the agent must return an empty list of anomalies.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect(dados)

    assert anomalias == [], (
        f"Expected no anomalies for insufficient data, got {len(anomalias)}: {anomalias}"
    )


# ---------------------------------------------------------------------------
# Property 5c: Required fields present in every anomaly record
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p5_required_fields_present(dados):
    """
    Property 5: Every anomaly record must contain required fields
    (subfuncao, tipo_indicador, ano, tipo_anomalia, descricao).

    **Validates: Requirements 6.3**

    Each anomaly must have all required fields with correct types,
    and tipo_anomalia must be one of the two valid anomaly types.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect(dados)

    for anom in anomalias:
        missing = REQUIRED_FIELDS - set(anom.keys())
        assert not missing, f"Missing required fields: {missing}"

        assert isinstance(anom["subfuncao"], int), (
            f"subfuncao should be int, got {type(anom['subfuncao'])}"
        )
        assert isinstance(anom["tipo_indicador"], str), (
            f"tipo_indicador should be str, got {type(anom['tipo_indicador'])}"
        )
        assert isinstance(anom["ano"], int), (
            f"ano should be int, got {type(anom['ano'])}"
        )
        assert isinstance(anom["tipo_anomalia"], str), (
            f"tipo_anomalia should be str, got {type(anom['tipo_anomalia'])}"
        )
        assert isinstance(anom["descricao"], str), (
            f"descricao should be str, got {type(anom['descricao'])}"
        )
        assert anom["tipo_anomalia"] in VALID_ANOMALY_TYPES, (
            f"tipo_anomalia '{anom['tipo_anomalia']}' not in {VALID_ANOMALY_TYPES}"
        )


# ---------------------------------------------------------------------------
# Property 5d: Only valid anomaly types are produced
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(dados=crossed_data_with_sufficient_pair())
def test_p5_only_valid_anomaly_types(dados):
    """
    Property 5: Every anomaly must have tipo_anomalia in
    {"alto_gasto_baixo_resultado", "baixo_gasto_alto_resultado"}.

    **Validates: Requirements 6.1, 6.2**

    No other anomaly types should be produced by the agent.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect(dados)

    for anom in anomalias:
        assert anom["tipo_anomalia"] in VALID_ANOMALY_TYPES, (
            f"Unexpected anomaly type: '{anom['tipo_anomalia']}'"
        )


# ---------------------------------------------------------------------------
# Property 5e: Empty input produces no anomalies
# ---------------------------------------------------------------------------


def test_p5_empty_input_produces_no_anomalies():
    """
    Property 5: Empty input must produce no anomalies.

    **Validates: Requirements 6.5**

    Edge case: when no data points are provided, the agent must return
    an empty list.
    """
    agent = AgenteAnomalias("prop5-anom")
    anomalias = agent.detect([])
    assert anomalias == [], (
        f"Expected no anomalies for empty input, got {len(anomalias)}"
    )
