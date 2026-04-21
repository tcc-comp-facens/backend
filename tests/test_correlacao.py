"""
Unit tests for AgenteCorrelacao.

Validates Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7.
"""

import math

from agents.base import AgenteBDI
from agents.analytical.correlacao import AgenteCorrelacao, _classify


class TestInheritanceAndInit:
    """Req 5.6: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        agent = AgenteCorrelacao("corr-test-1")
        assert isinstance(agent, AgenteBDI)

    def test_agent_id_set(self):
        agent = AgenteCorrelacao("corr-test-1")
        assert agent.agent_id == "corr-test-1"

    def test_initial_state_empty(self):
        agent = AgenteCorrelacao("corr-test-1")
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []

    def test_no_neo4j_required(self):
        """AgenteCorrelacao works on in-memory data, no neo4j_client needed."""
        agent = AgenteCorrelacao("corr-test-1")
        assert not hasattr(agent, "neo4j_client")


class TestBDICycle:
    """Req 5.6: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        agent = AgenteCorrelacao("corr-test-1")
        agent.update_beliefs({"dados_cruzados": [{"subfuncao": 301}]})
        perception = agent.perceive()
        assert "dados_cruzados" in perception
        assert len(perception["dados_cruzados"]) == 1

    def test_deliberate_creates_desire_with_data(self):
        agent = AgenteCorrelacao("corr-test-1")
        agent.update_beliefs({"dados_cruzados": [{"subfuncao": 301}]})
        desires = agent.deliberate()
        goals = [d["goal"] for d in desires]
        assert "calcular_correlacoes" in goals

    def test_deliberate_no_desires_without_data(self):
        agent = AgenteCorrelacao("corr-test-1")
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        agent = AgenteCorrelacao("corr-test-1")
        desires = [{"goal": "calcular_correlacoes"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 1
        assert intentions[0]["status"] == "pending"


class TestComputeBasic:
    """Req 5.5: Returns list with required fields."""

    def test_empty_input_returns_empty_list(self):
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute([])
        assert result == []

    def test_returns_required_fields(self):
        """Req 5.5: Each record has subfuncao, tipo_indicador, pearson,
        spearman, kendall, classificacao, n_pontos."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 200.0, 85.0),
            _point(301, "vacinacao", 2021, 300.0, 90.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)

        assert len(result) == 1
        record = result[0]
        required_fields = {
            "subfuncao", "tipo_indicador", "pearson", "spearman",
            "kendall", "classificacao", "n_pontos",
        }
        assert required_fields.issubset(set(record.keys()))

    def test_n_pontos_matches_data_count(self):
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 200.0, 85.0),
            _point(301, "vacinacao", 2021, 300.0, 90.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        assert result[0]["n_pontos"] == 3

    def test_subfuncao_and_tipo_preserved(self):
        dados = [
            _point(305, "dengue", 2019, 50.0, 1000.0),
            _point(305, "dengue", 2020, 60.0, 900.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        assert result[0]["subfuncao"] == 305
        assert result[0]["tipo_indicador"] == "dengue"


class TestCorrelationValues:
    """Reqs 5.1, 5.2, 5.3: Pearson, Spearman, Kendall calculations."""

    def test_perfect_positive_correlation(self):
        """Perfect linear positive: all correlations should be ~1.0."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 10.0),
            _point(301, "vacinacao", 2020, 200.0, 20.0),
            _point(301, "vacinacao", 2021, 300.0, 30.0),
            _point(301, "vacinacao", 2022, 400.0, 40.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["pearson"] == 1.0
        assert r["spearman"] == 1.0
        assert r["kendall"] == 1.0

    def test_perfect_negative_correlation(self):
        """Perfect linear negative: all correlations should be ~-1.0."""
        dados = [
            _point(302, "internacoes", 2019, 100.0, 40.0),
            _point(302, "internacoes", 2020, 200.0, 30.0),
            _point(302, "internacoes", 2021, 300.0, 20.0),
            _point(302, "internacoes", 2022, 400.0, 10.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["pearson"] == -1.0
        assert r["spearman"] == -1.0
        assert r["kendall"] == -1.0

    def test_correlations_in_valid_range(self):
        """All correlation values must be in [-1, 1]."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 150.0, 90.0),
            _point(301, "vacinacao", 2021, 120.0, 85.0),
            _point(301, "vacinacao", 2022, 200.0, 70.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert -1.0 <= r["pearson"] <= 1.0
        assert -1.0 <= r["spearman"] <= 1.0
        assert -1.0 <= r["kendall"] <= 1.0

    def test_multiple_pairs_computed_independently(self):
        """Each (subfuncao, tipo_indicador) pair is computed separately."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 200.0, 90.0),
            _point(305, "dengue", 2019, 50.0, 1000.0),
            _point(305, "dengue", 2020, 60.0, 800.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)

        assert len(result) == 2
        pairs = {(r["subfuncao"], r["tipo_indicador"]) for r in result}
        assert (301, "vacinacao") in pairs
        assert (305, "dengue") in pairs


class TestInsufficientData:
    """Req 5.7: Return 0.0 for all metrics when < 2 data points."""

    def test_single_point_returns_zeros(self):
        dados = [_point(301, "vacinacao", 2019, 100.0, 80.0)]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)

        assert len(result) == 1
        r = result[0]
        assert r["pearson"] == 0.0
        assert r["spearman"] == 0.0
        assert r["kendall"] == 0.0
        assert r["n_pontos"] == 1

    def test_single_point_classified_baixa(self):
        dados = [_point(301, "vacinacao", 2019, 100.0, 80.0)]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        assert result[0]["classificacao"] == "baixa"

    def test_mixed_sufficient_and_insufficient(self):
        """One pair with 1 point (zeros), another with 3 points (computed)."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(305, "dengue", 2019, 50.0, 1000.0),
            _point(305, "dengue", 2020, 60.0, 900.0),
            _point(305, "dengue", 2021, 70.0, 800.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)

        by_pair = {(r["subfuncao"], r["tipo_indicador"]): r for r in result}

        # Single point → zeros
        vac = by_pair[(301, "vacinacao")]
        assert vac["pearson"] == 0.0
        assert vac["spearman"] == 0.0
        assert vac["kendall"] == 0.0

        # Multiple points → computed (negative correlation expected)
        dengue = by_pair[(305, "dengue")]
        assert dengue["pearson"] != 0.0
        assert dengue["n_pontos"] == 3


class TestConstantValues:
    """Edge case: constant values should return 0.0 (scipy returns NaN)."""

    def test_constant_despesa_returns_zero(self):
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 100.0, 90.0),
            _point(301, "vacinacao", 2021, 100.0, 85.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["pearson"] == 0.0
        assert r["spearman"] == 0.0
        assert r["kendall"] == 0.0

    def test_constant_indicador_returns_zero(self):
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 200.0, 80.0),
            _point(301, "vacinacao", 2021, 300.0, 80.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["pearson"] == 0.0
        assert r["spearman"] == 0.0
        assert r["kendall"] == 0.0

    def test_both_constant_returns_zero(self):
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 100.0, 80.0),
            _point(301, "vacinacao", 2021, 100.0, 80.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["pearson"] == 0.0
        assert r["spearman"] == 0.0
        assert r["kendall"] == 0.0


class TestClassification:
    """Req 5.4: Classification based on Spearman coefficient."""

    def test_classify_alta_positive(self):
        assert _classify(0.7) == "alta"
        assert _classify(0.85) == "alta"
        assert _classify(1.0) == "alta"

    def test_classify_alta_negative(self):
        assert _classify(-0.7) == "alta"
        assert _classify(-0.85) == "alta"
        assert _classify(-1.0) == "alta"

    def test_classify_media(self):
        assert _classify(0.4) == "média"
        assert _classify(0.5) == "média"
        assert _classify(0.69) == "média"
        assert _classify(-0.4) == "média"
        assert _classify(-0.69) == "média"

    def test_classify_baixa(self):
        assert _classify(0.0) == "baixa"
        assert _classify(0.1) == "baixa"
        assert _classify(0.39) == "baixa"
        assert _classify(-0.39) == "baixa"

    def test_classification_uses_spearman(self):
        """Verify the agent uses Spearman (not Pearson) for classification."""
        # Perfect positive linear → Spearman = 1.0 → "alta"
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 10.0),
            _point(301, "vacinacao", 2020, 200.0, 20.0),
            _point(301, "vacinacao", 2021, 300.0, 30.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        assert result[0]["classificacao"] == "alta"


class TestTwoDataPoints:
    """Edge case: exactly 2 data points (minimum for correlation)."""

    def test_two_points_computes_correlation(self):
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 200.0, 90.0),
        ]
        agent = AgenteCorrelacao("corr-test-1")
        result = agent.compute(dados)
        r = result[0]

        assert r["n_pontos"] == 2
        # With 2 points, perfect positive linear → Pearson = 1.0
        assert r["pearson"] == 1.0
        assert r["spearman"] == 1.0
        assert r["kendall"] == 1.0


# -- Helpers ---------------------------------------------------------------

def _point(
    subfuncao: int,
    tipo_indicador: str,
    ano: int,
    valor_despesa: float,
    valor_indicador: float,
) -> dict:
    """Create a CrossedDataPoint dict for testing."""
    return {
        "subfuncao": subfuncao,
        "subfuncao_nome": f"Subfunção {subfuncao}",
        "tipo_indicador": tipo_indicador,
        "ano": ano,
        "valor_despesa": valor_despesa,
        "valor_indicador": valor_indicador,
    }
