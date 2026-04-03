"""Unit tests for MetricsCollector."""

import time
from unittest.mock import MagicMock

import pytest

from metrics import MetricsCollector


class TestMetricsCollector:
    """Tests for MetricsCollector start/stop/collect/persist semantics."""

    def test_collect_returns_required_fields(self):
        mc = MetricsCollector("agent-1", "consultor")
        mc.start()
        time.sleep(0.01)
        mc.stop()

        result = mc.collect()

        assert result["agentId"] == "agent-1"
        assert result["agentType"] == "consultor"
        assert isinstance(result["executionTimeMs"], int)
        assert result["executionTimeMs"] >= 10
        assert isinstance(result["cpuPercent"], float)
        assert isinstance(result["memoryMb"], float)
        assert result["memoryMb"] > 0

    def test_collect_without_start_raises(self):
        mc = MetricsCollector("agent-1", "consultor")
        with pytest.raises(RuntimeError, match="start"):
            mc.collect()

    def test_context_manager_sets_start_and_stop(self):
        with MetricsCollector("agent-2", "analisador") as mc:
            time.sleep(0.01)

        result = mc.collect()
        assert result["executionTimeMs"] >= 10
        assert result["agentId"] == "agent-2"
        assert result["agentType"] == "analisador"

    def test_collect_before_stop_uses_current_time(self):
        mc = MetricsCollector("agent-3", "orquestrador")
        mc.start()
        time.sleep(0.01)

        result = mc.collect()
        assert result["executionTimeMs"] >= 10

    def test_persist_calls_save_metrica_with_correct_shape(self):
        mock_client = MagicMock()

        mc = MetricsCollector("agent-4", "consultor")
        mc.start()
        mc.stop()

        result = mc.persist(mock_client, "analysis-123", "star")

        mock_client.save_metrica.assert_called_once()
        call_args = mock_client.save_metrica.call_args

        metrica = call_args[0][0]
        analysis_id = call_args[0][1]

        assert analysis_id == "analysis-123"
        assert metrica["architecture"] == "star"
        assert metrica["agentId"] == "agent-4"
        assert metrica["agentType"] == "consultor"
        assert "id" in metrica
        assert "recordedAt" in metrica
        assert "executionTimeMs" in metrica
        assert "cpuPercent" in metrica
        assert "memoryMb" in metrica

        # persist returns the same dict
        assert result is metrica

    def test_persist_generates_unique_ids(self):
        mock_client = MagicMock()

        mc = MetricsCollector("agent-5", "consultor")
        mc.start()
        mc.stop()

        r1 = mc.persist(mock_client, "a-1", "star")
        r2 = mc.persist(mock_client, "a-1", "star")

        assert r1["id"] != r2["id"]
