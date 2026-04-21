"""
Tests for the cross_domain_data utility function.

Validates Requirements: 9.4, 10.5
"""

from agents.data_crossing import (
    cross_domain_data,
    SUBFUNCAO_INDICADOR_MAP,
    MORTALIDADE_SUBFUNCOES,
)


class TestCrossDomainDataEmpty:
    """Handle empty inputs gracefully."""

    def test_empty_despesas(self):
        result = cross_domain_data([], [{"tipo": "dengue", "ano": 2020, "valor": 100.0}])
        assert result == []

    def test_empty_indicadores(self):
        result = cross_domain_data(
            [{"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 1000.0}],
            [],
        )
        assert result == []

    def test_both_empty(self):
        result = cross_domain_data([], [])
        assert result == []


class TestCrossDomainDataStandardMapping:
    """Standard subfunção→indicador mapping (301, 302, 305)."""

    def test_301_vacinacao(self):
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2020, "valor": 85.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 1
        assert result[0]["subfuncao"] == 301
        assert result[0]["tipo_indicador"] == "vacinacao"
        assert result[0]["ano"] == 2020
        assert result[0]["valor_despesa"] == 5000.0
        assert result[0]["valor_indicador"] == 85.0
        assert result[0]["subfuncao_nome"] == "Atenção Básica"

    def test_302_internacoes(self):
        despesas = [
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2021, "valor": 8000.0},
        ]
        indicadores = [
            {"tipo": "internacoes", "ano": 2021, "valor": 1200.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 1
        assert result[0]["subfuncao"] == 302
        assert result[0]["tipo_indicador"] == "internacoes"

    def test_305_dengue_and_covid(self):
        despesas = [
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 3000.0},
        ]
        indicadores = [
            {"tipo": "dengue", "ano": 2020, "valor": 500.0},
            {"tipo": "covid", "ano": 2020, "valor": 2000.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 2
        tipos = {r["tipo_indicador"] for r in result}
        assert tipos == {"dengue", "covid"}

    def test_no_matching_year(self):
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2021, "valor": 85.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert result == []

    def test_no_matching_subfuncao(self):
        despesas = [
            {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2020, "valor": 2000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2020, "valor": 85.0},
        ]
        # 303 is not in SUBFUNCAO_INDICADOR_MAP, and no mortalidade indicadores
        result = cross_domain_data(despesas, indicadores)
        assert result == []

    def test_multiple_years(self):
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2019, "valor": 4000.0},
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2021, "valor": 6000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2019, "valor": 80.0},
            {"tipo": "vacinacao", "ano": 2020, "valor": 85.0},
            {"tipo": "vacinacao", "ano": 2021, "valor": 90.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 3
        years = sorted(r["ano"] for r in result)
        assert years == [2019, 2020, 2021]


class TestCrossDomainDataMortalidade:
    """Mortalidade is transversal — crosses with ALL subfunções."""

    def test_mortalidade_crosses_all_subfuncoes(self):
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 8000.0},
            {"subfuncao": 303, "subfuncaoNome": "Suporte Profilático", "ano": 2020, "valor": 2000.0},
            {"subfuncao": 305, "subfuncaoNome": "Vigilância Epidemiológica", "ano": 2020, "valor": 3000.0},
        ]
        indicadores = [
            {"tipo": "mortalidade", "ano": 2020, "valor": 150.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 4
        subfuncoes = sorted(r["subfuncao"] for r in result)
        assert subfuncoes == [301, 302, 303, 305]
        for r in result:
            assert r["tipo_indicador"] == "mortalidade"
            assert r["valor_indicador"] == 150.0

    def test_mortalidade_only_matching_years(self):
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2021, "valor": 6000.0},
        ]
        indicadores = [
            {"tipo": "mortalidade", "ano": 2020, "valor": 150.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 1
        assert result[0]["ano"] == 2020

    def test_mortalidade_with_standard_indicators(self):
        """Mortalidade crossing coexists with standard mapping."""
        despesas = [
            {"subfuncao": 301, "subfuncaoNome": "Atenção Básica", "ano": 2020, "valor": 5000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2020, "valor": 85.0},
            {"tipo": "mortalidade", "ano": 2020, "valor": 150.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 2
        tipos = {r["tipo_indicador"] for r in result}
        assert tipos == {"vacinacao", "mortalidade"}


class TestCrossDomainDataOutputStructure:
    """Verify output structure matches CrossedDataPoint."""

    def test_all_required_fields_present(self):
        despesas = [
            {"subfuncao": 302, "subfuncaoNome": "Assistência Hospitalar", "ano": 2020, "valor": 8000.0},
        ]
        indicadores = [
            {"tipo": "internacoes", "ano": 2020, "valor": 1200.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 1
        point = result[0]
        required_keys = {"subfuncao", "subfuncao_nome", "tipo_indicador", "ano", "valor_despesa", "valor_indicador"}
        assert set(point.keys()) == required_keys

    def test_subfuncao_nome_fallback(self):
        """When subfuncaoNome is missing from despesa, use SUBFUNCAO_NOMES lookup."""
        despesas = [
            {"subfuncao": 301, "ano": 2020, "valor": 5000.0},
        ]
        indicadores = [
            {"tipo": "vacinacao", "ano": 2020, "valor": 85.0},
        ]
        result = cross_domain_data(despesas, indicadores)
        assert len(result) == 1
        assert result[0]["subfuncao_nome"] == "Atenção Básica"
