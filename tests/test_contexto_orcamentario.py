"""
Unit tests for AgenteContextoOrcamentario.

Validates Requirements 8.1, 8.2, 8.3, 8.4, 8.5.
"""

import math

from agents.base import AgenteBDI
from agents.context.contexto_orcamentario import (
    AgenteContextoOrcamentario,
    _classify_trend,
    _compute_yoy_variation,
)


# -- Helpers ---------------------------------------------------------------

def _despesa(subfuncao: int, ano: int, valor: float, nome: str = "") -> dict:
    """Create a DespesaRecord dict for testing."""
    if not nome:
        nomes = {
            301: "Atenção Básica",
            302: "Assistência Hospitalar",
            303: "Suporte Profilático",
            305: "Vigilância Epidemiológica",
        }
        nome = nomes.get(subfuncao, f"Subfunção {subfuncao}")
    return {
        "subfuncao": subfuncao,
        "subfuncaoNome": nome,
        "ano": ano,
        "valor": valor,
    }


# -- Inheritance and Init --------------------------------------------------

class TestInheritanceAndInit:
    """Req 8.4: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        assert isinstance(agent, AgenteBDI)

    def test_agent_id_set(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        assert agent.agent_id == "ctx-test-1"

    def test_initial_state_empty(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []

    def test_no_neo4j_required(self):
        """AgenteContextoOrcamentario works on in-memory data."""
        agent = AgenteContextoOrcamentario("ctx-test-1")
        assert not hasattr(agent, "neo4j_client")


# -- BDI Cycle --------------------------------------------------------------

class TestBDICycle:
    """Req 8.4: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        agent.update_beliefs({"despesas": [_despesa(301, 2019, 1000.0)]})
        perception = agent.perceive()
        assert "despesas" in perception
        assert len(perception["despesas"]) == 1

    def test_deliberate_creates_desire_with_data(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        agent.update_beliefs({"despesas": [_despesa(301, 2019, 1000.0)]})
        desires = agent.deliberate()
        goals = [d["goal"] for d in desires]
        assert "analisar_tendencias" in goals

    def test_deliberate_no_desires_without_data(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        desires = [{"goal": "analisar_tendencias"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 1
        assert intentions[0]["status"] == "pending"


# -- Empty and Insufficient Data --------------------------------------------

class TestEmptyAndInsufficientData:
    """Req 8.5: Handle empty input and insufficient data."""

    def test_empty_input_returns_empty_dict(self):
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends([])
        assert result == {}

    def test_single_year_returns_insuficiente(self):
        """Req 8.5: < 2 years → 'insuficiente'."""
        despesas = [_despesa(301, 2019, 1000.0)]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert 301 in result
        assert result[301]["tendencia"] == "insuficiente"
        assert result[301]["variacao_media_percentual"] == 0.0
        assert result[301]["anos_analisados"] == [2019]

    def test_single_year_multiple_subfuncoes(self):
        """Each subfunção with 1 year → all 'insuficiente'."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(302, 2020, 2000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "insuficiente"
        assert result[302]["tendencia"] == "insuficiente"


# -- Required Fields --------------------------------------------------------

class TestRequiredFields:
    """Req 8.3: Return dict with required fields."""

    def test_returns_required_fields(self):
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1400.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert 301 in result
        record = result[301]
        required_fields = {
            "subfuncao", "tendencia", "variacao_media_percentual", "anos_analisados",
        }
        assert required_fields.issubset(set(record.keys()))

    def test_insuficiente_has_required_fields(self):
        despesas = [_despesa(305, 2019, 500.0)]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        record = result[305]
        required_fields = {
            "subfuncao", "tendencia", "variacao_media_percentual", "anos_analisados",
        }
        assert required_fields.issubset(set(record.keys()))

    def test_anos_analisados_is_sorted(self):
        despesas = [
            _despesa(301, 2021, 1400.0),
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["anos_analisados"] == [2019, 2020, 2021]

    def test_result_keyed_by_subfuncao(self):
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(302, 2019, 2000.0),
            _despesa(302, 2020, 2500.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert set(result.keys()) == {301, 302}
        assert result[301]["subfuncao"] == 301
        assert result[302]["subfuncao"] == 302


# -- YoY Variation Computation (Req 8.1) ------------------------------------

class TestYoYVariation:
    """Req 8.1: Year-over-year percentage variation."""

    def test_basic_variation(self):
        """((1200 - 1000) / 1000) × 100 = 20%."""
        result = _compute_yoy_variation(1200.0, 1000.0)
        assert abs(result - 20.0) < 0.01

    def test_negative_variation(self):
        """((800 - 1000) / 1000) × 100 = -20%."""
        result = _compute_yoy_variation(800.0, 1000.0)
        assert abs(result - (-20.0)) < 0.01

    def test_zero_variation(self):
        """((1000 - 1000) / 1000) × 100 = 0%."""
        result = _compute_yoy_variation(1000.0, 1000.0)
        assert result == 0.0

    def test_division_by_zero_positive(self):
        """Previous = 0, current > 0 → +inf."""
        result = _compute_yoy_variation(100.0, 0.0)
        assert result == math.inf

    def test_division_by_zero_negative(self):
        """Previous = 0, current < 0 → -inf."""
        result = _compute_yoy_variation(-100.0, 0.0)
        assert result == -math.inf

    def test_division_by_zero_both_zero(self):
        """Previous = 0, current = 0 → 0%."""
        result = _compute_yoy_variation(0.0, 0.0)
        assert result == 0.0

    def test_known_variation_in_agent(self):
        """Verify the agent computes correct variation for known data."""
        # 2019→2020: ((1200-1000)/1000)*100 = 20%
        # 2020→2021: ((1440-1200)/1200)*100 = 20%
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1440.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert abs(result[301]["variacao_media_percentual"] - 20.0) < 0.01


# -- Trend Classification (Req 8.2) ----------------------------------------

class TestTrendClassification:
    """Req 8.2: Classify trends as crescimento, corte, estagnacao."""

    def test_crescimento_consecutive_positive(self):
        """Consecutive positive variations for 2+ years → crescimento."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1500.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "crescimento"

    def test_corte_consecutive_negative(self):
        """Consecutive negative variations for 2+ years → corte."""
        despesas = [
            _despesa(301, 2019, 1500.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "corte"

    def test_estagnacao_small_variations(self):
        """All |variation| < 5% → estagnacao."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1020.0),  # +2%
            _despesa(301, 2021, 1010.0),  # ~-1%
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "estagnacao"

    def test_two_years_positive_is_crescimento(self):
        """Exactly 2 years with positive variation → crescimento."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1500.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "crescimento"

    def test_two_years_negative_is_corte(self):
        """Exactly 2 years with negative variation → corte."""
        despesas = [
            _despesa(301, 2019, 1500.0),
            _despesa(301, 2020, 1000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "corte"

    def test_two_years_stagnation(self):
        """Exactly 2 years with |variation| < 5% → estagnacao."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1030.0),  # +3%
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "estagnacao"


# -- Division by Zero Edge Case ---------------------------------------------

class TestDivisionByZero:
    """Edge case: valor_n-1 = 0."""

    def test_zero_to_positive_is_crescimento(self):
        """Previous year = 0, current > 0 → infinite growth → crescimento."""
        despesas = [
            _despesa(301, 2019, 0.0),
            _despesa(301, 2020, 1000.0),
            _despesa(301, 2021, 2000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "crescimento"

    def test_zero_to_zero_is_estagnacao(self):
        """Both years = 0 → 0% variation → estagnacao."""
        despesas = [
            _despesa(301, 2019, 0.0),
            _despesa(301, 2020, 0.0),
            _despesa(301, 2021, 0.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "estagnacao"


# -- Multiple Subfunções ----------------------------------------------------

class TestMultipleSubfuncoes:
    """Multiple subfunções analyzed independently."""

    def test_independent_subfuncao_analysis(self):
        despesas = [
            # 301: crescimento
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1500.0),
            # 302: corte
            _despesa(302, 2019, 2000.0),
            _despesa(302, 2020, 1500.0),
            _despesa(302, 2021, 1000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "crescimento"
        assert result[302]["tendencia"] == "corte"

    def test_mixed_sufficient_and_insufficient(self):
        """One subfunção with 1 year (insuficiente), another with 3 years."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(302, 2019, 2000.0),
            _despesa(302, 2020, 2500.0),
            _despesa(302, 2021, 3000.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["tendencia"] == "insuficiente"
        assert result[302]["tendencia"] == "crescimento"


# -- Variação Média Percentual ---------------------------------------------

class TestVariacaoMedia:
    """Verify variacao_media_percentual computation."""

    def test_uniform_growth(self):
        """Uniform 20% growth each year → average = 20%."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 1200.0),
            _despesa(301, 2021, 1440.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert abs(result[301]["variacao_media_percentual"] - 20.0) < 0.01

    def test_uniform_decline(self):
        """Uniform 20% decline each year → average = -20%."""
        despesas = [
            _despesa(301, 2019, 1000.0),
            _despesa(301, 2020, 800.0),
            _despesa(301, 2021, 640.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert abs(result[301]["variacao_media_percentual"] - (-20.0)) < 0.01

    def test_insuficiente_has_zero_variacao(self):
        despesas = [_despesa(301, 2019, 1000.0)]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        assert result[301]["variacao_media_percentual"] == 0.0


# -- Classify Trend Helper --------------------------------------------------

class TestClassifyTrendHelper:
    """Unit tests for the _classify_trend helper function."""

    def test_empty_variations(self):
        assert _classify_trend([]) == "insuficiente"

    def test_all_positive(self):
        assert _classify_trend([10.0, 15.0, 20.0]) == "crescimento"

    def test_all_negative(self):
        assert _classify_trend([-10.0, -15.0, -20.0]) == "corte"

    def test_all_small(self):
        assert _classify_trend([1.0, -2.0, 3.0]) == "estagnacao"

    def test_single_positive_large(self):
        """Single variation > 5% but only 1 year → not enough for streak."""
        result = _classify_trend([50.0])
        assert result == "crescimento"

    def test_single_negative_large(self):
        result = _classify_trend([-50.0])
        assert result == "corte"

    def test_mixed_with_positive_streak(self):
        """Negative then two positives → crescimento (streak of 2)."""
        assert _classify_trend([-10.0, 20.0, 15.0]) == "crescimento"

    def test_mixed_with_negative_streak(self):
        """Positive then two negatives → corte (streak of 2)."""
        assert _classify_trend([10.0, -20.0, -15.0]) == "corte"


# -- Duplicate Year Aggregation ---------------------------------------------

class TestDuplicateYearAggregation:
    """Multiple despesa entries for the same subfunção and year are summed."""

    def test_aggregates_same_year(self):
        despesas = [
            _despesa(301, 2019, 500.0),
            _despesa(301, 2019, 500.0),  # same year → sum = 1000
            _despesa(301, 2020, 1200.0),
        ]
        agent = AgenteContextoOrcamentario("ctx-test-1")
        result = agent.analyze_trends(despesas)

        # (1200 - 1000) / 1000 * 100 = 20%
        assert abs(result[301]["variacao_media_percentual"] - 20.0) < 0.01
