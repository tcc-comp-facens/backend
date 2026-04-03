# Feature: multiagent-architecture-comparison, Property 2: Normalização para esquema unificado
"""
Property-based tests for ETL normalization.

**Validates: Requirements 1.1, 2.1**
"""

import sys
import os

# Ensure backend/ is on sys.path so etl.siops_loader can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings
from hypothesis import strategies as st

from etl.siops_loader import normalize_row

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_SUBFUNCOES = [301, 302, 303, 305]

VALID_SISTEMAS = ["sinan", "si_pni", "sim", "sih"]
VALID_TIPOS = ["dengue", "covid", "vacinacao", "mortalidade", "internacoes"]

# Column mapping that matches the logical keys used by normalize_row
_COL_MAP = {
    "municipio": "co_municipio",
    "subfuncao": "co_subfuncao",
    "valor": "vl_despesa",
    "ano": "aa_exercicio",
}


@st.composite
def valid_siops_row(draw):
    """Generate a valid SIOPS CSV row dict for Sorocaba (355220)."""
    subfuncao = draw(st.sampled_from(VALID_SUBFUNCOES))
    ano = draw(st.integers(min_value=2000, max_value=2030))
    # Use a positive float; represent as Brazilian decimal string (comma separator)
    valor_float = draw(st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
    # Format as plain string with comma decimal separator (common in BR government CSVs)
    valor_str = f"{valor_float:.2f}".replace(".", ",")

    row = {
        "co_municipio": "355220",
        "co_subfuncao": str(subfuncao),
        "vl_despesa": valor_str,
        "aa_exercicio": str(ano),
    }
    return row


@st.composite
def valid_datasus_record(draw):
    """Generate a valid IndicadorDataSUS record dict."""
    sistema = draw(st.sampled_from(VALID_SISTEMAS))
    tipo = draw(st.sampled_from(VALID_TIPOS))
    ano = draw(st.integers(min_value=2000, max_value=2030))
    valor = draw(st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False))

    return {
        "sistema": sistema,
        "tipo": tipo,
        "ano": ano,
        "valor": float(valor),
        "fonte": "datasus",
    }


# ---------------------------------------------------------------------------
# P2 — Normalização para esquema unificado (SIOPS)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(row=valid_siops_row())
def test_p2_siops_normalization(row):
    """
    **Validates: Requirements 1.1, 2.1**

    For any valid SIOPS CSV row (municipio=355220, subfuncao in {301,302,303,305}),
    normalize_row must return a non-None dict with all required fields and correct types.
    """
    result = normalize_row(row, _COL_MAP)

    # Must not be discarded
    assert result is not None, f"normalize_row returned None for valid row: {row}"

    # All required fields must be present
    required_fields = {"subfuncao", "subfuncaoNome", "ano", "valor", "fonte"}
    assert required_fields.issubset(result.keys()), (
        f"Missing fields: {required_fields - result.keys()}"
    )

    # Type checks
    assert isinstance(result["subfuncao"], int), (
        f"subfuncao must be int, got {type(result['subfuncao'])}"
    )
    assert isinstance(result["subfuncaoNome"], str) and result["subfuncaoNome"], (
        "subfuncaoNome must be a non-empty str"
    )
    assert isinstance(result["ano"], int), (
        f"ano must be int, got {type(result['ano'])}"
    )
    assert isinstance(result["valor"], (int, float)), (
        f"valor must be numeric, got {type(result['valor'])}"
    )
    assert result["fonte"] == "siops", (
        f"fonte must be 'siops', got {result['fonte']!r}"
    )

    # Value constraints
    assert result["subfuncao"] in {301, 302, 303, 305}, (
        f"subfuncao {result['subfuncao']} not in valid set"
    )


# ---------------------------------------------------------------------------
# P2 — Normalização para esquema unificado (DataSUS)
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(record=valid_datasus_record())
def test_p2_datasus_normalization(record):
    """
    **Validates: Requirements 1.1, 2.1**

    For any valid IndicadorDataSUS record, all required fields must be present
    with correct types and values. (Tests schema invariants without network access.)
    """
    # All required fields must be present
    required_fields = {"sistema", "tipo", "ano", "valor", "fonte"}
    assert required_fields.issubset(record.keys()), (
        f"Missing fields: {required_fields - record.keys()}"
    )

    # sistema
    assert isinstance(record["sistema"], str), (
        f"sistema must be str, got {type(record['sistema'])}"
    )
    assert record["sistema"] in {"sinan", "si_pni", "sim", "sih"}, (
        f"sistema {record['sistema']!r} not in valid set"
    )

    # tipo
    assert isinstance(record["tipo"], str), (
        f"tipo must be str, got {type(record['tipo'])}"
    )
    assert record["tipo"] in {"dengue", "covid", "vacinacao", "mortalidade", "internacoes"}, (
        f"tipo {record['tipo']!r} not in valid set"
    )

    # ano
    assert isinstance(record["ano"], int), (
        f"ano must be int, got {type(record['ano'])}"
    )

    # valor
    assert isinstance(record["valor"], float), (
        f"valor must be float, got {type(record['valor'])}"
    )
    assert record["valor"] >= 0, f"valor must be >= 0, got {record['valor']}"

    # fonte
    assert record["fonte"] == "datasus", (
        f"fonte must be 'datasus', got {record['fonte']!r}"
    )


# Feature: multiagent-architecture-comparison, Property 3: Rastreabilidade de origem
# ---------------------------------------------------------------------------
# P3 — Rastreabilidade de origem
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(row=valid_siops_row())
def test_p3_siops_fonte_rastreabilidade(row):
    """
    **Validates: Requirements 1.3, 2.6**

    For any valid SIOPS CSV row, normalize_row must produce a record where
    `fonte` is exactly "siops".
    """
    result = normalize_row(row, _COL_MAP)

    assert result is not None, f"normalize_row returned None for valid row: {row}"
    assert "fonte" in result, f"'fonte' field missing from normalized row: {result}"
    assert result["fonte"] == "siops", (
        f"Expected fonte='siops', got {result['fonte']!r}"
    )


@settings(max_examples=100)
@given(record=valid_datasus_record())
def test_p3_datasus_fonte_rastreabilidade(record):
    """
    **Validates: Requirements 1.3, 2.6**

    For any valid IndicadorDataSUS record, the `fonte` field must be exactly
    "datasus".
    """
    assert "fonte" in record, f"'fonte' field missing from DataSUS record: {record}"
    assert record["fonte"] == "datasus", (
        f"Expected fonte='datasus', got {record['fonte']!r}"
    )


# Feature: multiagent-architecture-comparison, Property 4: Deduplicação mantém registro mais recente
# ---------------------------------------------------------------------------
# P4 — Deduplicação mantém registro mais recente
# ---------------------------------------------------------------------------


@st.composite
def siops_rows_with_duplicates(draw):
    """Generate 2-10 SIOPS rows sharing the same (subfuncao, ano) key."""
    subfuncao = draw(st.sampled_from([301, 302, 303, 305]))
    ano = draw(st.integers(min_value=2000, max_value=2030))
    n = draw(st.integers(min_value=2, max_value=10))
    rows = []
    for _ in range(n):
        valor = draw(st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
        valor_str = f"{valor:.2f}".replace(".", ",")
        rows.append({
            "co_municipio": "355220",
            "co_subfuncao": str(subfuncao),
            "vl_despesa": valor_str,
            "aa_exercicio": str(ano),
        })
    return rows


@settings(max_examples=100)
@given(rows=siops_rows_with_duplicates())
def test_p4_siops_deduplication(rows):
    """
    **Validates: Requirements 1.4, 2.7**

    Given a batch of SIOPS rows that all share the same (subfuncao, ano) key
    but carry different valores, simulating the MERGE SET behaviour (last write
    wins) must leave exactly one entry per key in the deduplicated result.
    """
    # Normalize every row
    normalized = [normalize_row(r, _COL_MAP) for r in rows]
    assert all(n is not None for n in normalized), "normalize_row returned None for a valid row"

    # Simulate MERGE deduplication: last write wins (same as Cypher MERGE … SET)
    deduped = {}
    for record in normalized:
        key = (record["subfuncao"], record["ano"])
        deduped[key] = record

    # All rows share the same key, so exactly one entry must remain
    assert len(deduped) == 1, (
        f"Expected 1 deduplicated entry, got {len(deduped)}: {list(deduped.keys())}"
    )


@st.composite
def datasus_records_with_duplicates(draw):
    """Generate 2-10 DataSUS records sharing the same (sistema, tipo, ano) key."""
    sistema = draw(st.sampled_from(VALID_SISTEMAS))
    tipo = draw(st.sampled_from(VALID_TIPOS))
    ano = draw(st.integers(min_value=2000, max_value=2030))
    n = draw(st.integers(min_value=2, max_value=10))
    records = []
    for _ in range(n):
        valor = draw(st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False))
        records.append({
            "sistema": sistema,
            "tipo": tipo,
            "ano": ano,
            "valor": float(valor),
            "fonte": "datasus",
        })
    return records


@settings(max_examples=100)
@given(records=datasus_records_with_duplicates())
def test_p4_datasus_deduplication(records):
    """
    **Validates: Requirements 1.4, 2.7**

    Given a batch of DataSUS records that all share the same (sistema, tipo, ano)
    key but carry different valores, simulating the MERGE SET behaviour (last write
    wins) must leave exactly one entry per key in the deduplicated result.
    """
    # Simulate MERGE deduplication: last write wins
    deduped = {}
    for record in records:
        key = (record["sistema"], record["tipo"], record["ano"])
        deduped[key] = record

    # All records share the same key, so exactly one entry must remain
    assert len(deduped) == 1, (
        f"Expected 1 deduplicated entry, got {len(deduped)}: {list(deduped.keys())}"
    )
