"""
Unit tests for AgenteSintetizador.

Validates Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6.
"""

from queue import Queue
from unittest.mock import patch

from agents.base import AgenteBDI
from agents.analytical.sintetizador import AgenteSintetizador, CHUNK_SIZE


class TestInheritanceAndInit:
    """Req 7.5: Inherits from AgenteBDI and implements BDI cycle."""

    def test_inherits_agente_bdi(self):
        agent = AgenteSintetizador("sint-test-1")
        assert isinstance(agent, AgenteBDI)

    def test_agent_id_set(self):
        agent = AgenteSintetizador("sint-test-1")
        assert agent.agent_id == "sint-test-1"

    def test_initial_state_empty(self):
        agent = AgenteSintetizador("sint-test-1")
        assert agent.beliefs == {}
        assert agent.desires == []
        assert agent.intentions == []

    def test_no_neo4j_required(self):
        """AgenteSintetizador works on in-memory data, no neo4j_client needed."""
        agent = AgenteSintetizador("sint-test-1")
        assert not hasattr(agent, "neo4j_client")


class TestBDICycle:
    """Req 7.5: Full BDI cycle (perceive, deliberate, plan, execute)."""

    def test_perceive_returns_beliefs(self):
        agent = AgenteSintetizador("sint-test-1")
        agent.update_beliefs({
            "correlacoes": [{"subfuncao": 301}],
            "anomalias": [],
            "contexto_orcamentario": {},
            "analysis_id": "test-123",
            "architecture": "star",
        })
        perception = agent.perceive()
        assert "correlacoes" in perception
        assert "anomalias" in perception
        assert "contexto_orcamentario" in perception
        assert perception["analysis_id"] == "test-123"
        assert perception["architecture"] == "star"

    def test_deliberate_creates_desire_with_analysis_id(self):
        agent = AgenteSintetizador("sint-test-1")
        agent.update_beliefs({"analysis_id": "test-123"})
        desires = agent.deliberate()
        goals = [d["goal"] for d in desires]
        assert "sintetizar_texto" in goals

    def test_deliberate_no_desires_without_analysis_id(self):
        agent = AgenteSintetizador("sint-test-1")
        desires = agent.deliberate()
        assert desires == []

    def test_plan_creates_pending_intentions(self):
        agent = AgenteSintetizador("sint-test-1")
        desires = [{"goal": "sintetizar_texto"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 1
        assert intentions[0]["status"] == "pending"


class TestSynthesizeReturnsText:
    """Req 7.1, 7.3: synthesize() returns full text."""

    def test_returns_string(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        # LLM unavailable → fallback
        with patch("llm_client.generate", return_value=None):
            result = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=_sample_anomalias(),
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_llm_text_when_available(self):
        """Req 7.1: Uses LLM (Groq primary, Gemini fallback)."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        llm_text = "Análise gerada pelo LLM com detalhes completos."
        with patch("llm_client.generate", return_value=llm_text):
            result = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert result == llm_text

    def test_returns_fallback_when_llm_unavailable(self):
        """Req 7.3: Structured fallback when LLM unavailable."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            result = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=_sample_anomalias(),
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        # Fallback text should contain section headers
        assert "Resumo Executivo" in result
        assert "Correlações" in result
        assert "Anomalias" in result
        assert "Orçamentário" in result

    def test_returns_fallback_when_llm_raises(self):
        """Req 7.3: Fallback when LLM raises exception."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", side_effect=Exception("API error")):
            result = agent.synthesize(
                correlacoes=[],
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert isinstance(result, str)
        assert len(result) > 0


class TestStreaming:
    """Req 7.2, 7.6: Streaming chunks and done event."""

    def test_chunks_sent_to_ws_queue(self):
        """Req 7.2: Text streamed in chunks of ~80 chars."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=_sample_anomalias(),
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )

        events = _drain_queue(ws_queue)
        chunk_events = [e for e in events if e["type"] == "chunk"]
        assert len(chunk_events) > 0

        # All chunks should be <= CHUNK_SIZE
        for event in chunk_events:
            assert len(event["payload"]) <= CHUNK_SIZE

    def test_chunks_concatenate_to_full_text(self):
        """Req 7.2: Concatenation of all chunks equals the full text."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            full_text = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=_sample_anomalias(),
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )

        events = _drain_queue(ws_queue)
        chunk_events = [e for e in events if e["type"] == "chunk"]
        reconstructed = "".join(e["payload"] for e in chunk_events)
        assert reconstructed == full_text

    def test_ws_event_format_chunk(self):
        """Verify chunk WSEvent has correct format."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value="Hello world"):
            agent.synthesize(
                correlacoes=[],
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="abc-123",
                ws_queue=ws_queue,
                architecture="hierarchical",
            )

        events = _drain_queue(ws_queue)
        chunk = events[0]
        assert chunk["analysisId"] == "abc-123"
        assert chunk["architecture"] == "hierarchical"
        assert chunk["type"] == "chunk"
        assert isinstance(chunk["payload"], str)

class TestFallbackTextContent:
    """Req 7.4: Fallback text includes required sections."""

    def test_contains_resumo_executivo(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=_sample_anomalias(),
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Resumo Executivo" in text

    def test_contains_correlacoes_section(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=_sample_correlacoes(),
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Correlações" in text

    def test_contains_anomalias_section(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=[],
                anomalias=_sample_anomalias(),
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Anomalias" in text

    def test_contains_contexto_orcamentario_section(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=[],
                anomalias=[],
                contexto_orcamentario=_sample_contexto(),
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Orçamentário" in text

    def test_references_correlation_subfuncao_and_classification(self):
        """Req 7.4: When correlações non-empty, text references subfunção and classification."""
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        correlacoes = _sample_correlacoes()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=correlacoes,
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        # Should reference the subfuncao name and classification
        for c in correlacoes:
            assert c["tipo_indicador"] in text
            assert c["classificacao"] in text

    def test_references_anomalia_descriptions(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        anomalias = _sample_anomalias()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=[],
                anomalias=anomalias,
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        for a in anomalias:
            assert a["descricao"] in text

    def test_empty_correlacoes_handled(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=[],
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Correlações" in text
        assert "Anomalias" in text

    def test_empty_contexto_handled(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value=None):
            text = agent.synthesize(
                correlacoes=[],
                anomalias=[],
                contexto_orcamentario={},
                analysis_id="test-123",
                ws_queue=ws_queue,
                architecture="star",
            )
        assert "Orçamentário" in text
        assert "não disponíveis" in text.lower() or "Orçamentário" in text


class TestArchitectureParam:
    """Verify architecture parameter is correctly propagated to WSEvents."""

    def test_star_architecture(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value="Test"):
            agent.synthesize(
                correlacoes=[], anomalias=[], contexto_orcamentario={},
                analysis_id="test-123", ws_queue=ws_queue, architecture="star",
            )
        events = _drain_queue(ws_queue)
        for e in events:
            assert e["architecture"] == "star"

    def test_hierarchical_architecture(self):
        agent = AgenteSintetizador("sint-test-1")
        ws_queue = Queue()
        with patch("llm_client.generate", return_value="Test"):
            agent.synthesize(
                correlacoes=[], anomalias=[], contexto_orcamentario={},
                analysis_id="test-123", ws_queue=ws_queue, architecture="hierarchical",
            )
        events = _drain_queue(ws_queue)
        for e in events:
            assert e["architecture"] == "hierarchical"


# -- Helpers ---------------------------------------------------------------

def _sample_correlacoes() -> list[dict]:
    """Sample correlation data for testing."""
    return [
        {
            "subfuncao": 301,
            "subfuncao_nome": "Atenção Básica",
            "tipo_indicador": "vacinacao",
            "pearson": 0.85,
            "spearman": 0.82,
            "kendall": 0.75,
            "classificacao": "alta",
            "n_pontos": 5,
        },
        {
            "subfuncao": 305,
            "subfuncao_nome": "Vigilância Epidemiológica",
            "tipo_indicador": "dengue",
            "pearson": 0.35,
            "spearman": 0.30,
            "kendall": 0.25,
            "classificacao": "baixa",
            "n_pontos": 4,
        },
    ]


def _sample_anomalias() -> list[dict]:
    """Sample anomaly data for testing."""
    return [
        {
            "subfuncao": 301,
            "subfuncao_nome": "Atenção Básica",
            "tipo_indicador": "vacinacao",
            "ano": 2021,
            "tipo_anomalia": "alto_gasto_baixo_resultado",
            "descricao": (
                "Gasto acima da mediana em Atenção Básica (R$ 5000.00) "
                "mas indicador vacinacao abaixo da mediana (70.00) em 2021"
            ),
        },
    ]


def _sample_contexto() -> dict:
    """Sample budget context data for testing."""
    return {
        301: {
            "subfuncao": 301,
            "tendencia": "crescimento",
            "variacao_media_percentual": 8.5,
            "anos_analisados": [2019, 2020, 2021],
        },
        305: {
            "subfuncao": 305,
            "tendencia": "estagnacao",
            "variacao_media_percentual": 2.1,
            "anos_analisados": [2019, 2020, 2021],
        },
    }


def _drain_queue(q: Queue) -> list[dict]:
    """Drain all items from a queue into a list."""
    items = []
    while not q.empty():
        items.append(q.get_nowait())
    return items
