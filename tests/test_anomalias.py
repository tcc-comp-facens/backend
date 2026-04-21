"""
Unit tests for AgenteAnomalias.

Validates Requirements 6.1, 6.2, 6.3, 6.4, 6.5.
"""

from agents.base import AgenteBDI
from agents.analytical.anomalias import AgenteAnomalias, _median


class TestInheritanceAndInit:
    """Req 6.4: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        agent = AgenteAnomalias("anom-test-1")
        assert isinstance(agent, AgenteBDI)

    def test_agent_id_set(self):
        agent = AgenteAnomalias("anom-test-1")
        assert agent.agent_id == "anom-test-1"

    def test_initial_state_empty(self):
        agent = AgenteAnomalias("anom-test-1")
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []

    def test_no_neo4j_required(self):
        """AgenteAnomalias works on in-memory data, no neo4j_client needed."""
        agent = AgenteAnomalias("anom-test-1")
        assert not hasattr(agent, "neo4j_client")


class TestBDICycle:
    """Req 6.4: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        agent = AgenteAnomalias("anom-test-1")
        agent.update_beliefs({"dados_cruzados": [{"subfuncao": 301}]})
        perception = agent.perceive()
        assert "dados_cruzados" in perception
        assert len(perception["dados_cruzados"]) == 1

    def test_deliberate_creates_desire_with_data(self):
        agent = AgenteAnomalias("anom-test-1")
        agent.update_beliefs({"dados_cruzados": [{"subfuncao": 301}]})
        desires = agent.deliberate()
        goals = [d["goal"] for d in desires]
        assert "detectar_anomalias" in goals

    def test_deliberate_no_desires_without_data(self):
        agent = AgenteAnomalias("anom-test-1")
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        agent = AgenteAnomalias("anom-test-1")
        desires = [{"goal": "detectar_anomalias"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 1
        assert intentions[0]["status"] == "pending"


class TestDetectBasic:
    """Req 6.3: Returns list with required fields."""

    def test_empty_input_returns_empty_list(self):
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect([])
        assert result == []

    def test_returns_required_fields(self):
        """Req 6.3: Each record has subfuncao, tipo_indicador, ano,
        tipo_anomalia, descricao."""
        dados = [
            _point(301, "vacinacao", 2019, 5000.0, 90.0),
            _point(301, "vacinacao", 2020, 1000.0, 50.0),
            _point(301, "vacinacao", 2021, 6000.0, 40.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        assert len(result) > 0
        required_fields = {
            "subfuncao", "tipo_indicador", "ano", "tipo_anomalia", "descricao",
        }
        for record in result:
            assert required_fields.issubset(set(record.keys()))

    def test_subfuncao_and_tipo_preserved(self):
        dados = [
            _point(305, "dengue", 2019, 5000.0, 100.0),
            _point(305, "dengue", 2020, 1000.0, 500.0),
            _point(305, "dengue", 2021, 6000.0, 50.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        for record in result:
            assert record["subfuncao"] == 305
            assert record["tipo_indicador"] == "dengue"


class TestAltoGastoBaixoResultado:
    """Req 6.1: Detect high spending with low outcome."""

    def test_detects_alto_gasto_baixo_resultado(self):
        """When despesa > median AND indicador < median → anomaly."""
        # median despesa = 3000, median indicador = 80
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        alto_gasto = [r for r in result if r["tipo_anomalia"] == "alto_gasto_baixo_resultado"]
        assert len(alto_gasto) == 1
        assert alto_gasto[0]["ano"] == 2021
        assert alto_gasto[0]["subfuncao"] == 301

    def test_alto_gasto_descricao_in_portuguese(self):
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        alto_gasto = [r for r in result if r["tipo_anomalia"] == "alto_gasto_baixo_resultado"]
        assert len(alto_gasto) == 1
        desc = alto_gasto[0]["descricao"]
        assert "2021" in desc
        assert "mediana" in desc.lower()


class TestBaixoGastoAltoResultado:
    """Req 6.2: Detect low spending with high outcome."""

    def test_detects_baixo_gasto_alto_resultado(self):
        """When despesa < median AND indicador > median → anomaly."""
        # median despesa = 3000, median indicador = 80
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        baixo_gasto = [r for r in result if r["tipo_anomalia"] == "baixo_gasto_alto_resultado"]
        assert len(baixo_gasto) == 1
        assert baixo_gasto[0]["ano"] == 2019
        assert baixo_gasto[0]["subfuncao"] == 301

    def test_baixo_gasto_descricao_in_portuguese(self):
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        baixo_gasto = [r for r in result if r["tipo_anomalia"] == "baixo_gasto_alto_resultado"]
        assert len(baixo_gasto) == 1
        desc = baixo_gasto[0]["descricao"]
        assert "2019" in desc
        assert "mediana" in desc.lower()


class TestInsufficientData:
    """Req 6.5: Ignore pairs with < 2 data points."""

    def test_single_point_returns_no_anomalies(self):
        dados = [_point(301, "vacinacao", 2019, 5000.0, 10.0)]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)
        assert result == []

    def test_mixed_sufficient_and_insufficient(self):
        """One pair with 1 point (ignored), another with 3 points (analyzed)."""
        dados = [
            _point(301, "vacinacao", 2019, 5000.0, 10.0),
            _point(305, "dengue", 2019, 1000.0, 500.0),
            _point(305, "dengue", 2020, 3000.0, 300.0),
            _point(305, "dengue", 2021, 5000.0, 100.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        # Only dengue pair should produce anomalies
        for r in result:
            assert r["subfuncao"] == 305
            assert r["tipo_indicador"] == "dengue"


class TestNoAnomalies:
    """Edge case: data at the median produces no anomalies."""

    def test_all_equal_values_no_anomalies(self):
        """When all values are equal, nothing is above or below median."""
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 80.0),
            _point(301, "vacinacao", 2020, 100.0, 80.0),
            _point(301, "vacinacao", 2021, 100.0, 80.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)
        assert result == []

    def test_values_at_median_no_anomalies(self):
        """Points exactly at the median are not flagged (strict inequality)."""
        # median despesa = 200, median indicador = 80
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 70.0),
            _point(301, "vacinacao", 2020, 200.0, 80.0),
            _point(301, "vacinacao", 2021, 300.0, 90.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        # 2019: despesa < median, indicador < median → not anomaly
        # 2020: despesa == median, indicador == median → not anomaly (strict >/<)
        # 2021: despesa > median, indicador > median → not anomaly
        assert result == []


class TestMultiplePairs:
    """Multiple (subfuncao, tipo_indicador) pairs analyzed independently."""

    def test_independent_pair_analysis(self):
        dados = [
            # Pair 1: 301/vacinacao — should detect anomalies
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
            # Pair 2: 305/dengue — all equal, no anomalies
            _point(305, "dengue", 2019, 100.0, 50.0),
            _point(305, "dengue", 2020, 100.0, 50.0),
            _point(305, "dengue", 2021, 100.0, 50.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        # Only pair 1 should produce anomalies
        for r in result:
            assert r["subfuncao"] == 301
            assert r["tipo_indicador"] == "vacinacao"


class TestMedianHelper:
    """Unit tests for the _median helper function."""

    def test_odd_length(self):
        assert _median([1.0, 2.0, 3.0]) == 2.0

    def test_even_length(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_unsorted_input(self):
        assert _median([3.0, 1.0, 2.0]) == 2.0

    def test_two_elements(self):
        assert _median([10.0, 20.0]) == 15.0

    def test_single_element(self):
        assert _median([5.0]) == 5.0


class TestTwoDataPoints:
    """Edge case: exactly 2 data points (minimum for anomaly detection)."""

    def test_two_points_can_detect_anomalies(self):
        """With 2 points, median is the average. Points above/below can be flagged."""
        # median despesa = 150, median indicador = 55
        dados = [
            _point(301, "vacinacao", 2019, 100.0, 60.0),
            _point(301, "vacinacao", 2020, 200.0, 50.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        # 2019: despesa(100) < median(150), indicador(60) > median(55) → baixo_gasto_alto_resultado
        # 2020: despesa(200) > median(150), indicador(50) < median(55) → alto_gasto_baixo_resultado
        assert len(result) == 2
        tipos = {r["tipo_anomalia"] for r in result}
        assert "alto_gasto_baixo_resultado" in tipos
        assert "baixo_gasto_alto_resultado" in tipos


class TestDescricaoFormat:
    """Verify descricao field contains human-readable Portuguese text."""

    def test_descricao_contains_subfuncao_info(self):
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        for r in result:
            assert "301" in r["descricao"] or "Subfunção" in r["descricao"]
            assert str(r["ano"]) in r["descricao"]

    def test_descricao_contains_r_currency(self):
        dados = [
            _point(301, "vacinacao", 2019, 1000.0, 90.0),
            _point(301, "vacinacao", 2020, 3000.0, 80.0),
            _point(301, "vacinacao", 2021, 5000.0, 70.0),
        ]
        agent = AgenteAnomalias("anom-test-1")
        result = agent.detect(dados)

        for r in result:
            assert "R$" in r["descricao"]


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
