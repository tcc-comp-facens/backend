"""
Tests for AgenteAnalisadorStar.

Covers correlation calculation, classification, anomaly detection,
BDI cycle, streaming to ws_queue, and the analyze() public API.
"""

from __future__ import annotations

import math
from queue import Queue

import pytest

from agents.star.analyzer import (
    AgenteAnalisadorStar,
    SUBFUNCAO_INDICADOR_MAP,
    _classify,
    _pearson,
)


# ---------------------------------------------------------------------------
# _pearson helper
# ---------------------------------------------------------------------------

class TestPearson:
    def test_perfect_positive(self):
        assert _pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)

    def test_perfect_negative(self):
        assert _pearson([1, 2, 3], [6, 4, 2]) == pytest.approx(-1.0)

    def test_no_correlation_returns_near_zero(self):
        # orthogonal-ish data
        r = _pearson([1, 2, 3, 4], [1, -1, 1, -1])
        assert -0.5 < r < 0.5

    def test_fewer_than_two_points(self):
        assert _pearson([1], [2]) == 0.0
        assert _pearson([], []) == 0.0

    def test_constant_values_return_zero(self):
        assert _pearson([5, 5, 5], [1, 2, 3]) == 0.0

    def test_result_clamped(self):
        r = _pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
        assert -1.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# _classify helper
# ---------------------------------------------------------------------------

class TestClassify:
    def test_alta_positive(self):
        assert _classify(0.7) == "alta"
        assert _classify(0.95) == "alta"

    def test_alta_negative(self):
        assert _classify(-0.7) == "alta"
        assert _classify(-1.0) == "alta"

    def test_media(self):
        assert _classify(0.4) == "média"
        assert _classify(0.69) == "média"
        assert _classify(-0.5) == "média"

    def test_baixa(self):
        assert _classify(0.0) == "baixa"
        assert _classify(0.39) == "baixa"
        assert _classify(-0.1) == "baixa"


# ---------------------------------------------------------------------------
# Helpers to build test data
# ---------------------------------------------------------------------------

def _make_despesas(subfuncao: int, years_values: list[tuple[int, float]]) -> list[dict]:
    return [
        {"subfuncao": subfuncao, "ano": y, "valor": v}
        for y, v in years_values
    ]


def _make_indicadores(tipo: str, years_values: list[tuple[int, float]]) -> list[dict]:
    return [
        {"tipo": tipo, "ano": y, "valor": v}
        for y, v in years_values
    ]


# ---------------------------------------------------------------------------
# AgenteAnalisadorStar — cross data
# ---------------------------------------------------------------------------

class TestCrossData:
    def test_crosses_matching_subfuncao_and_tipo(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "despesas": _make_despesas(301, [(2020, 100), (2021, 200)]),
            "indicadores": _make_indicadores("vacinacao", [(2020, 80), (2021, 90)]),
        })
        agent._cross_data()
        crossed = agent.beliefs["dados_cruzados"]
        assert len(crossed) == 2
        assert all(c["subfuncao"] == 301 for c in crossed)
        assert all(c["tipo_indicador"] == "vacinacao" for c in crossed)

    def test_ignores_unmatched_subfuncao(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "despesas": _make_despesas(303, [(2020, 100)]),  # 303 not in map
            "indicadores": _make_indicadores("vacinacao", [(2020, 80)]),
        })
        agent._cross_data()
        assert agent.beliefs["dados_cruzados"] == []

    def test_only_common_years(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "despesas": _make_despesas(301, [(2020, 100), (2022, 300)]),
            "indicadores": _make_indicadores("vacinacao", [(2020, 80), (2021, 90)]),
        })
        agent._cross_data()
        crossed = agent.beliefs["dados_cruzados"]
        assert len(crossed) == 1
        assert crossed[0]["ano"] == 2020


# ---------------------------------------------------------------------------
# AgenteAnalisadorStar — correlations
# ---------------------------------------------------------------------------

class TestComputeCorrelations:
    def test_produces_valid_correlation(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "dados_cruzados": [
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 100, "valor_indicador": 80, "ano": 2020},
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 200, "valor_indicador": 90, "ano": 2021},
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 300, "valor_indicador": 95, "ano": 2022},
            ],
        })
        agent._compute_correlations()
        corrs = agent.beliefs["correlacoes"]
        assert len(corrs) == 1
        assert -1.0 <= corrs[0]["correlacao"] <= 1.0
        assert corrs[0]["classificacao"] in ("alta", "média", "baixa")

    def test_empty_data_produces_no_correlations(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({"dados_cruzados": []})
        agent._compute_correlations()
        assert agent.beliefs["correlacoes"] == []


# ---------------------------------------------------------------------------
# AgenteAnalisadorStar — anomalies
# ---------------------------------------------------------------------------

class TestDetectAnomalies:
    def test_detects_high_spend_low_outcome(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "dados_cruzados": [
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 100, "valor_indicador": 90, "ano": 2020},
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 500, "valor_indicador": 10, "ano": 2021},
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 150, "valor_indicador": 85, "ano": 2022},
            ],
        })
        agent._detect_anomalies()
        anomalias = agent.beliefs["anomalias"]
        types = [a["tipo_anomalia"] for a in anomalias]
        assert "alto_gasto_baixo_resultado" in types

    def test_no_anomalies_with_single_point(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        agent.update_beliefs({
            "dados_cruzados": [
                {"subfuncao": 301, "tipo_indicador": "vacinacao", "valor_despesa": 100, "valor_indicador": 80, "ano": 2020},
            ],
        })
        agent._detect_anomalies()
        assert agent.beliefs["anomalias"] == []


# ---------------------------------------------------------------------------
# AgenteAnalisadorStar — streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_streams_chunks_and_done(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        ws_queue: Queue = Queue()
        agent.update_beliefs({
            "analysis_id": "test-123",
            "ws_queue": ws_queue,
            "correlacoes": [],
            "anomalias": [],
        })
        agent._stream_analysis()

        events = []
        while not ws_queue.empty():
            events.append(ws_queue.get_nowait())

        assert len(events) >= 2  # at least one chunk + done
        assert all(e["analysisId"] == "test-123" for e in events)
        assert all(e["architecture"] == "star" for e in events)

        chunk_events = [e for e in events if e["type"] == "chunk"]
        done_events = [e for e in events if e["type"] == "done"]
        assert len(chunk_events) >= 1
        assert len(done_events) == 1
        assert done_events[0] == events[-1]  # done is last

    def test_chunks_reconstruct_full_text(self):
        agent = AgenteAnalisadorStar("test-analyzer")
        ws_queue: Queue = Queue()
        agent.update_beliefs({
            "analysis_id": "test-456",
            "ws_queue": ws_queue,
            "correlacoes": [
                {"subfuncao": 301, "subfuncao_nome": "Atenção Básica", "tipo_indicador": "vacinacao",
                 "correlacao": 0.85, "classificacao": "alta", "n_pontos": 5},
            ],
            "anomalias": [],
        })
        agent._stream_analysis()

        full_text = agent.beliefs["texto_analise"]
        chunks = []
        while not ws_queue.empty():
            e = ws_queue.get_nowait()
            if e["type"] == "chunk":
                chunks.append(e["payload"])

        assert "".join(chunks) == full_text


# ---------------------------------------------------------------------------
# AgenteAnalisadorStar — analyze() public API
# ---------------------------------------------------------------------------

class TestAnalyzeAPI:
    def test_full_pipeline(self):
        ws_queue: Queue = Queue()
        agent = AgenteAnalisadorStar("test-analyzer")

        data = {
            "despesas": _make_despesas(301, [(2020, 100), (2021, 200), (2022, 300)]),
            "indicadores": _make_indicadores("vacinacao", [(2020, 80), (2021, 85), (2022, 95)]),
        }

        result = agent.analyze("analysis-789", data, ws_queue)

        assert "correlacoes" in result
        assert "anomalias" in result
        assert "texto_analise" in result
        assert len(result["correlacoes"]) >= 1
        assert isinstance(result["texto_analise"], str)
        assert len(result["texto_analise"]) > 0

        # ws_queue should have events
        events = []
        while not ws_queue.empty():
            events.append(ws_queue.get_nowait())
        assert events[-1]["type"] == "done"

    def test_empty_data_still_completes(self):
        ws_queue: Queue = Queue()
        agent = AgenteAnalisadorStar("test-analyzer")
        result = agent.analyze("analysis-empty", {"despesas": [], "indicadores": []}, ws_queue)
        assert result["correlacoes"] == []
        assert result["anomalias"] == []
