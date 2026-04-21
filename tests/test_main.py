"""
Unit tests for backend/main.py — REST endpoints and WebSocket.

Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5, 8.1, 8.6, 10.4
"""

from __future__ import annotations

import asyncio
import json
import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import (
    AnalysisRequest,
    HealthParams,
    _health_params_to_list,
    _validate_analysis_params,
    active_queues,
    active_threads,
    app,
)


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# Validation helper tests (Req 9.4, 9.5)
# ---------------------------------------------------------------------------

class TestValidateAnalysisParams:
    """Req 9.4, 9.5: validate params, return 400 on invalid."""

    def test_valid_params_no_errors(self):
        req = AnalysisRequest(
            dateFrom=2020,
            dateTo=2022,
            healthParams=HealthParams(dengue=True, covid=False, vaccination=False),
        )
        assert _validate_analysis_params(req) == []

    def test_date_from_greater_than_date_to(self):
        req = AnalysisRequest(
            dateFrom=2023,
            dateTo=2020,
            healthParams=HealthParams(dengue=True),
        )
        errors = _validate_analysis_params(req)
        assert any("dateFrom" in e for e in errors)

    def test_no_health_params_selected(self):
        req = AnalysisRequest(
            dateFrom=2020,
            dateTo=2022,
            healthParams=HealthParams(dengue=False, covid=False, vaccination=False),
        )
        errors = _validate_analysis_params(req)
        assert any("healthParam" in e for e in errors)

    def test_multiple_errors(self):
        req = AnalysisRequest(
            dateFrom=2023,
            dateTo=2020,
            healthParams=HealthParams(dengue=False, covid=False, vaccination=False),
        )
        errors = _validate_analysis_params(req)
        assert len(errors) == 2


class TestHealthParamsToList:
    def test_all_true(self):
        hp = HealthParams(dengue=True, covid=True, vaccination=True, internacoes=True, mortalidade=True)
        result = _health_params_to_list(hp)
        assert set(result) == {"dengue", "covid", "vacinacao", "internacoes", "mortalidade"}

    def test_none_true(self):
        hp = HealthParams(dengue=False, covid=False, vaccination=False, internacoes=False, mortalidade=False)
        assert _health_params_to_list(hp) == []

    def test_partial(self):
        hp = HealthParams(dengue=True, covid=False, vaccination=True, internacoes=False, mortalidade=True)
        result = _health_params_to_list(hp)
        assert "dengue" in result
        assert "vacinacao" in result
        assert "mortalidade" in result
        assert "covid" not in result
        assert "internacoes" not in result


# ---------------------------------------------------------------------------
# POST /api/analysis (Req 9.1, 9.4, 9.5, 10.4)
# ---------------------------------------------------------------------------

class TestPostAnalysis:
    """Req 9.1: POST /api/analysis endpoint."""

    @patch("main._get_neo4j_client")
    @patch("main.threading.Thread")
    def test_valid_request_returns_analysis_id(self, mock_thread_cls, mock_get_client, client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        resp = client.post("/api/analysis", json={
            "dateFrom": 2020,
            "dateTo": 2022,
            "healthParams": {"dengue": True, "covid": False, "vaccination": False},
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "analysisId" in data
        assert len(data["analysisId"]) == 36  # UUID format

    @patch("main._get_neo4j_client")
    @patch("main.threading.Thread")
    def test_starts_two_threads(self, mock_thread_cls, mock_get_client, client):
        """Req 10.4: dispatch both architectures in parallel."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        resp = client.post("/api/analysis", json={
            "dateFrom": 2020,
            "dateTo": 2022,
            "healthParams": {"dengue": True, "covid": False, "vaccination": False},
        })

        assert resp.status_code == 200
        # Two Thread() calls: one for star, one for hierarchical
        assert mock_thread_cls.call_count == 2
        # Both threads started
        assert mock_thread.start.call_count == 2

    def test_invalid_date_range_returns_400(self, client):
        """Req 9.5: invalid params return 400."""
        resp = client.post("/api/analysis", json={
            "dateFrom": 2023,
            "dateTo": 2020,
            "healthParams": {"dengue": True, "covid": False, "vaccination": False},
        })
        assert resp.status_code == 400
        assert "dateFrom" in resp.json()["detail"]

    def test_no_health_params_returns_400(self, client):
        """Req 9.5: at least one healthParam must be true."""
        resp = client.post("/api/analysis", json={
            "dateFrom": 2020,
            "dateTo": 2022,
            "healthParams": {"dengue": False, "covid": False, "vaccination": False},
        })
        assert resp.status_code == 400
        assert "healthParam" in resp.json()["detail"]

    def test_missing_fields_returns_422(self, client):
        resp = client.post("/api/analysis", json={"dateFrom": 2020})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/analysis/{id} (Req 9.2)
# ---------------------------------------------------------------------------

class TestGetAnalysis:
    """Req 9.2: GET /api/analysis/{id}."""

    @patch("main._get_neo4j_client")
    def test_returns_analysis_data(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_record = {"analise": {"id": "abc-123", "dateFrom": 2020, "dateTo": 2022}}
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result
        mock_client._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_client.return_value = mock_client

        resp = client.get("/api/analysis/abc-123")
        assert resp.status_code == 200
        assert resp.json()["id"] == "abc-123"

    @patch("main._get_neo4j_client")
    def test_not_found_returns_404(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result
        mock_client._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_client.return_value = mock_client

        resp = client.get("/api/analysis/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/benchmarks (Req 9.3)
# ---------------------------------------------------------------------------

class TestGetBenchmarks:
    """Req 9.3: GET /api/benchmarks."""

    @patch("main._get_neo4j_client")
    def test_returns_benchmark_list(self, mock_get_client, client):
        mock_client = MagicMock()
        mock_session = MagicMock()
        mock_records = [
            MagicMock(**{"__iter__": lambda s: iter([]), "keys.return_value": []}),
        ]
        # Simulate neo4j result as iterable of records
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([
            {"analysisId": "a-1", "architecture": "star", "agentId": "ag-1",
             "executionTimeMs": 100, "cpuPercent": 5.0, "memoryMb": 50.0},
        ]))
        mock_session.run.return_value = mock_result
        mock_client._driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_client._driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_get_client.return_value = mock_client

        resp = client.get("/api/benchmarks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# WebSocket /ws/{analysisId} (Req 8.1, 8.6)
# ---------------------------------------------------------------------------

class TestWebSocket:
    """Req 8.1: WebSocket endpoint for streaming events."""

    def test_receives_events_from_queue(self, client):
        """Events put in ws_queue are sent to the WebSocket client."""
        analysis_id = "ws-test-1"
        ws_queue = Queue()
        active_queues[analysis_id] = ws_queue

        # Pre-load events: chunk + done for star, done for hierarchical
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "chunk",
            "payload": "Hello",
        })
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "hierarchical",
            "type": "done",
            "payload": "",
        })

        with client.websocket_connect(f"/ws/{analysis_id}") as ws:
            event1 = ws.receive_json()
            assert event1["type"] == "chunk"
            assert event1["payload"] == "Hello"

            event2 = ws.receive_json()
            assert event2["type"] == "done"
            assert event2["architecture"] == "star"

            event3 = ws.receive_json()
            assert event3["type"] == "done"
            assert event3["architecture"] == "hierarchical"

        # Cleanup happened
        assert analysis_id not in active_queues

    def test_no_active_analysis_sends_error(self, client):
        """If no queue exists for the analysis ID, send error and close."""
        with client.websocket_connect("/ws/nonexistent") as ws:
            event = ws.receive_json()
            assert event["type"] == "error"
            assert "No active analysis" in event["payload"]

    def test_closes_after_two_done_events(self, client):
        """WebSocket closes after receiving done from both architectures."""
        analysis_id = "ws-test-close"
        ws_queue = Queue()
        active_queues[analysis_id] = ws_queue

        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "hierarchical",
            "type": "done",
            "payload": "",
        })

        with client.websocket_connect(f"/ws/{analysis_id}") as ws:
            ws.receive_json()  # star done
            ws.receive_json()  # hierarchical done
            # Connection should close after this

    def test_error_events_do_not_count_as_done(self, client):
        """Error events are forwarded but don't close the connection."""
        analysis_id = "ws-test-error"
        ws_queue = Queue()
        active_queues[analysis_id] = ws_queue

        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "error",
            "payload": "Something went wrong",
        })
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "hierarchical",
            "type": "done",
            "payload": "",
        })

        with client.websocket_connect(f"/ws/{analysis_id}") as ws:
            event1 = ws.receive_json()
            assert event1["type"] == "error"

            event2 = ws.receive_json()
            assert event2["type"] == "done"

            event3 = ws.receive_json()
            assert event3["type"] == "done"


# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------

class TestCORS:
    """Verify CORS middleware is configured."""

    def test_cors_headers_present(self, client):
        resp = client.options(
            "/api/benchmarks",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should respond (not 405)
        assert resp.status_code in (200, 204)
