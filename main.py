"""
FastAPI application — REST endpoints and WebSocket for the
multiagent architecture comparison system.

Endpoints:
  POST /api/analysis       — Start a new analysis (star + hierarchical in parallel)
  GET  /api/analysis/{id}  — Retrieve analysis result from Neo4j
  GET  /api/benchmarks     — Comparative metrics across analyses
  WS   /ws/{analysisId}    — Real-time streaming of agent events

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 8.1, 8.6, 10.4
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timezone
from queue import Empty, Queue
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.star.orchestrator import OrquestradorEstrela
from agents.hierarchical.coordinator import CoordenadorGeral
from db.neo4j_client import Neo4jClient
from quality_metrics import compute_all_quality_metrics, generate_comparative_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Multiagent Architecture Comparison")

# ---------------------------------------------------------------------------
# CORS — accept requests from the frontend (Req 10.4)
# ---------------------------------------------------------------------------
origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Shared state — active queues and threads per analysis
# ---------------------------------------------------------------------------
active_queues: dict[str, Queue] = {}
active_threads: dict[str, list[threading.Thread]] = {}
active_results: dict[str, dict[str, Any]] = {}  # analysisId → {"star": result, "hierarchical": result}
active_agent_metrics: dict[str, dict[str, list[dict]]] = {}  # analysisId → {"star": [...], "hierarchical": [...]}


# ---------------------------------------------------------------------------
# Neo4j client helper
# ---------------------------------------------------------------------------
def _get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class HealthParams(BaseModel):
    dengue: bool = False
    covid: bool = False
    vaccination: bool = False
    internacoes: bool = False
    mortalidade: bool = False


class AnalysisRequest(BaseModel):
    dateFrom: int = 2019
    dateTo: int = 2021
    healthParams: HealthParams


class AnalysisResponse(BaseModel):
    analysisId: str


# ---------------------------------------------------------------------------
# Validation helper (Req 9.4, 9.5)
# ---------------------------------------------------------------------------
def _validate_analysis_params(req: AnalysisRequest) -> list[str]:
    """Return a list of validation error messages (empty == valid)."""
    errors: list[str] = []
    if req.dateFrom > req.dateTo:
        errors.append("dateFrom must be <= dateTo")
    hp = req.healthParams
    if not (hp.dengue or hp.covid or hp.vaccination or hp.internacoes or hp.mortalidade):
        errors.append("At least one healthParam must be true")
    return errors


# ---------------------------------------------------------------------------
# Thread runners
# ---------------------------------------------------------------------------
def _health_params_to_list(hp: HealthParams) -> list[str]:
    """Convert HealthParams booleans to a list of type strings."""
    params: list[str] = []
    if hp.dengue:
        params.append("dengue")
    if hp.covid:
        params.append("covid")
    if hp.vaccination:
        params.append("vacinacao")
    if hp.internacoes:
        params.append("internacoes")
    if hp.mortalidade:
        params.append("mortalidade")
    return params


def _persist_topology_result(
    neo4j_client: Neo4jClient,
    analysis_id: str,
    architecture: str,
    message_count: int,
    texto_analise: str,
) -> None:
    """Persist topology completion metadata in Neo4j (Req 11.3, 11.4).

    Updates the Analise node with message count, text analysis, status,
    and completion timestamp for the given architecture.
    """
    completed_at = datetime.now(timezone.utc).isoformat()

    if architecture == "star":
        query = """
        MATCH (a:Analise {id: $analysisId})
        SET a.starStatus       = 'completed',
            a.starMessageCount = $messageCount,
            a.starTextAnalysis = $textoAnalise,
            a.starCompletedAt  = $completedAt
        """
    else:
        query = """
        MATCH (a:Analise {id: $analysisId})
        SET a.hierStatus       = 'completed',
            a.hierMessageCount = $messageCount,
            a.hierTextAnalysis = $textoAnalise,
            a.hierCompletedAt  = $completedAt
        """

    with neo4j_client._driver.session() as session:
        session.run(
            query,
            analysisId=analysis_id,
            messageCount=message_count,
            textoAnalise=texto_analise,
            completedAt=completed_at,
        )


def _run_star(analysis_id: str, params: dict[str, Any], ws_queue: Queue) -> None:
    """Execute the star architecture pipeline in a dedicated thread."""
    neo4j_client: Neo4jClient | None = None
    try:
        neo4j_client = _get_neo4j_client()
        orchestrator = OrquestradorEstrela(
            agent_id=f"star-orch-{uuid.uuid4().hex[:8]}",
            neo4j_client=neo4j_client,
        )
        result = orchestrator.run(analysis_id, params, ws_queue)

        # Store result for quality metrics computation
        if analysis_id not in active_results:
            active_results[analysis_id] = {}
        active_results[analysis_id]["star"] = result

        # Persist message count and analysis text in Neo4j (Req 11.3)
        _persist_topology_result(
            neo4j_client,
            analysis_id,
            architecture="star",
            message_count=result.get("message_count", 0),
            texto_analise=result.get("texto_analise", ""),
        )

        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
    except Exception as exc:
        logger.error("Star thread failed: %s", exc)
        # OrquestradorEstrela already sends error event; add done sentinel
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "star",
            "type": "done",
            "payload": "",
        })
    finally:
        try:
            if neo4j_client is not None:
                neo4j_client.close()
        except Exception:
            pass


def _run_hierarchical(analysis_id: str, params: dict[str, Any], ws_queue: Queue) -> None:
    """Execute the hierarchical architecture pipeline in a dedicated thread."""
    neo4j_client: Neo4jClient | None = None
    try:
        neo4j_client = _get_neo4j_client()
        coordinator = CoordenadorGeral(
            agent_id=f"hier-coord-{uuid.uuid4().hex[:8]}",
            neo4j_client=neo4j_client,
        )
        result = coordinator.run(analysis_id, params, ws_queue)

        # Store result for quality metrics computation
        if analysis_id not in active_results:
            active_results[analysis_id] = {}
        active_results[analysis_id]["hierarchical"] = result

        # Persist message count and analysis text in Neo4j (Req 11.3)
        _persist_topology_result(
            neo4j_client,
            analysis_id,
            architecture="hierarchical",
            message_count=result.get("message_count", 0),
            texto_analise=result.get("texto_analise", ""),
        )

        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "hierarchical",
            "type": "done",
            "payload": "",
        })
    except Exception as exc:
        logger.error("Hierarchical thread failed: %s", exc)
        ws_queue.put({
            "analysisId": analysis_id,
            "architecture": "hierarchical",
            "type": "done",
            "payload": "",
        })
    finally:
        try:
            if neo4j_client is not None:
                neo4j_client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/analysis", response_model=AnalysisResponse)
async def create_analysis(req: AnalysisRequest):
    """Start a new analysis — validates params, creates record, launches threads.

    Req 9.1: POST /api/analysis
    Req 9.4, 9.5: Validate params, return 400 on invalid
    Req 10.4: Dispatch both architectures in parallel
    """
    # Validate
    errors = _validate_analysis_params(req)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    analysis_id = str(uuid.uuid4())
    health_list = _health_params_to_list(req.healthParams)

    # Persist initial analysis record in Neo4j
    neo4j_client = _get_neo4j_client()
    try:
        neo4j_client.save_analise({
            "id": analysis_id,
            "dateFrom": req.dateFrom,
            "dateTo": req.dateTo,
            "healthParams": req.healthParams.model_dump(),
            "starStatus": "pending",
            "hierStatus": "pending",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        })

        # Link existing DespesaSIOPS nodes to this analysis
        with neo4j_client._driver.session() as session:
            session.run(
                """
                MATCH (a:Analise {id: $id}), (d:DespesaSIOPS)
                WHERE d.ano >= $dateFrom AND d.ano <= $dateTo
                MERGE (a)-[:POSSUI_DESPESA]->(d)
                """,
                id=analysis_id,
                dateFrom=req.dateFrom,
                dateTo=req.dateTo,
            )
            # Link existing IndicadorDataSUS nodes to this analysis
            session.run(
                """
                MATCH (a:Analise {id: $id}), (i:IndicadorDataSUS)
                WHERE i.ano >= $dateFrom AND i.ano <= $dateTo
                  AND i.tipo IN $healthParams
                MERGE (a)-[:POSSUI_INDICADOR]->(i)
                """,
                id=analysis_id,
                dateFrom=req.dateFrom,
                dateTo=req.dateTo,
                healthParams=health_list,
            )
    finally:
        neo4j_client.close()

    # Shared queue for WebSocket streaming
    ws_queue: Queue = Queue()
    active_queues[analysis_id] = ws_queue

    params: dict[str, Any] = {
        "date_from": req.dateFrom,
        "date_to": req.dateTo,
        "health_params": health_list,
    }

    # Launch two threads in parallel (Req 10.4)
    t_star = threading.Thread(
        target=_run_star,
        args=(analysis_id, params, ws_queue),
        daemon=True,
    )
    t_hier = threading.Thread(
        target=_run_hierarchical,
        args=(analysis_id, params, ws_queue),
        daemon=True,
    )
    active_threads[analysis_id] = [t_star, t_hier]
    t_star.start()
    t_hier.start()

    return AnalysisResponse(analysisId=analysis_id)


@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve analysis result from Neo4j.

    Req 9.2: GET /api/analysis/{id}
    """
    neo4j_client = _get_neo4j_client()
    try:
        query = """
        MATCH (a:Analise {id: $analysisId})
        RETURN a {.*} AS analise
        """
        with neo4j_client._driver.session() as session:
            result = session.run(query, analysisId=analysis_id)
            record = result.single()
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis not found")
        return record["analise"]
    finally:
        neo4j_client.close()


@app.get("/api/benchmarks")
async def get_benchmarks():
    """Return comparative metrics from Neo4j.

    Req 9.3: GET /api/benchmarks
    """
    neo4j_client = _get_neo4j_client()
    try:
        query = """
        MATCH (a:Analise)-[:GEROU_METRICA]->(m:MetricaExecucao)
        RETURN a.id AS analysisId,
               m.architecture AS architecture,
               m.agentId AS agentId,
               m.executionTimeMs AS executionTimeMs,
               m.cpuPercent AS cpuPercent,
               m.memoryMb AS memoryMb
        ORDER BY a.createdAt DESC, m.architecture, m.agentId
        """
        with neo4j_client._driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    finally:
        neo4j_client.close()


@app.get("/api/analysis/{analysis_id}/quality")
async def get_quality_metrics(analysis_id: str):
    """Return quality and efficiency metrics for a completed analysis.

    Computes metrics across three axes:
    - Efficiency: coordination overhead, latency breakdown, communication efficiency
    - Quality: deterministic consistency, faithfulness, completeness, structural quality
    - Resilience: partial result coverage

    Query param ?llm_judge=true enables LLM-as-judge faithfulness evaluation.
    """
    from fastapi import Query

    results = active_results.get(analysis_id, {})
    star_result = results.get("star")
    hier_result = results.get("hierarchical")

    # Return cached quality metrics if already computed
    if "quality_metrics" in results:
        return results["quality_metrics"]

    if not star_result or not hier_result:
        raise HTTPException(
            status_code=404,
            detail="Quality metrics not available. Both topologies must complete first.",
        )

    # Reconstruct agent metrics from stored results
    # (they may not be in active_agent_metrics if WebSocket wasn't connected)
    star_agent_metrics = results.get("star_agent_metrics", [])
    hier_agent_metrics = results.get("hier_agent_metrics", [])

    quality = compute_all_quality_metrics(
        star_result=star_result,
        hier_result=hier_result,
        star_agent_metrics=star_agent_metrics,
        hier_agent_metrics=hier_agent_metrics,
        star_message_count=star_result.get("message_count", 0),
        hier_message_count=hier_result.get("message_count", 0),
        use_llm_judge=False,
    )

    # Cache for subsequent requests
    active_results[analysis_id]["quality_metrics"] = quality
    return quality


@app.get("/api/analysis/{analysis_id}/report")
async def get_comparative_report(analysis_id: str):
    """Return the comparative textual report for a completed analysis.

    Returns the full text of the comparative report between star and
    hierarchical topologies, including efficiency, quality, and resilience.
    """
    results = active_results.get(analysis_id, {})

    # Return cached report if available
    if "comparative_report" in results:
        return {"report": results["comparative_report"]}

    # Try to generate it
    star_result = results.get("star")
    hier_result = results.get("hierarchical")
    if not star_result or not hier_result:
        raise HTTPException(
            status_code=404,
            detail="Report not available. Both topologies must complete first.",
        )

    quality = results.get("quality_metrics")
    if not quality:
        raise HTTPException(
            status_code=404,
            detail="Quality metrics not computed yet. Access /quality first.",
        )

    report = generate_comparative_report(
        quality=quality,
        star_agent_metrics=[],
        hier_agent_metrics=[],
        star_message_count=star_result.get("message_count", 0),
        hier_message_count=hier_result.get("message_count", 0),
    )
    active_results[analysis_id]["comparative_report"] = report
    return {"report": report}


# ---------------------------------------------------------------------------
# WebSocket Endpoint (Req 8.1, 8.6)
# ---------------------------------------------------------------------------

@app.websocket("/ws/{analysis_id}")
async def websocket_endpoint(websocket: WebSocket, analysis_id: str):
    """Stream events from the shared ws_queue to the client.

    Events: chunk, done, error, metric.
    Closes when both architectures have sent 'done' or 'error'.
    On client disconnect, cancels threads and cleans up (Req 8.6).
    """
    await websocket.accept()

    ws_queue = active_queues.get(analysis_id)
    if ws_queue is None:
        await websocket.send_json({
            "analysisId": analysis_id,
            "architecture": "",
            "type": "error",
            "payload": "No active analysis found for this ID",
        })
        await websocket.close()
        return

    done_count = 0
    loop = asyncio.get_event_loop()
    captured_agent_metrics: dict[str, list[dict]] = {"star": [], "hierarchical": []}
    captured_message_counts: dict[str, int] = {"star": 0, "hierarchical": 0}

    try:
        while done_count < 2:
            try:
                event = await loop.run_in_executor(None, lambda: ws_queue.get(timeout=1.0))
            except Empty:
                continue

            event_type = event.get("type", "?")
            event_arch = event.get("architecture", "?")
            logger.info(
                "WS %s: sending event type=%s arch=%s (done_count=%d)",
                analysis_id[:8], event_type, event_arch, done_count,
            )

            # Capture agent metrics from metric events for quality computation
            if event_type == "metric" and isinstance(event.get("payload"), dict):
                payload = event["payload"]
                arch = payload.get("architecture", "")
                if arch in captured_agent_metrics:
                    captured_agent_metrics[arch] = payload.get("agentMetrics", [])
                    captured_message_counts[arch] = payload.get("messageCount", 0)

            await websocket.send_json(event)

            if event_type == "done":
                done_count += 1
                logger.info(
                    "WS %s: done_count now %d", analysis_id[:8], done_count,
                )

        # Both topologies done — compute quality metrics if results available
        results = active_results.get(analysis_id, {})
        star_result = results.get("star", {})
        hier_result = results.get("hierarchical", {})

        if star_result and hier_result:
            try:
                quality = compute_all_quality_metrics(
                    star_result=star_result,
                    hier_result=hier_result,
                    star_agent_metrics=captured_agent_metrics.get("star", []),
                    hier_agent_metrics=captured_agent_metrics.get("hierarchical", []),
                    star_message_count=captured_message_counts.get("star", 0),
                    hier_message_count=captured_message_counts.get("hierarchical", 0),
                    use_llm_judge=False,
                )
                # Send quality metrics as a special event
                await websocket.send_json({
                    "analysisId": analysis_id,
                    "architecture": "both",
                    "type": "quality_metrics",
                    "payload": quality,
                })
                # Store for the REST endpoint
                active_results[analysis_id]["quality_metrics"] = quality

                # Generate and stream comparative report
                report = generate_comparative_report(
                    quality=quality,
                    star_agent_metrics=captured_agent_metrics.get("star", []),
                    hier_agent_metrics=captured_agent_metrics.get("hierarchical", []),
                    star_message_count=captured_message_counts.get("star", 0),
                    hier_message_count=captured_message_counts.get("hierarchical", 0),
                    data_coverage=star_result.get("data_coverage"),
                )
                active_results[analysis_id]["comparative_report"] = report

                # Stream report in chunks
                chunk_size = 80
                for i in range(0, len(report), chunk_size):
                    chunk = report[i: i + chunk_size]
                    await websocket.send_json({
                        "analysisId": analysis_id,
                        "architecture": "both",
                        "type": "chunk",
                        "payload": chunk,
                    })
                # Signal report done
                await websocket.send_json({
                    "analysisId": analysis_id,
                    "architecture": "both",
                    "type": "done",
                    "payload": "",
                })

                logger.info("WS %s: comparative report sent (%d chars)", analysis_id[:8], len(report))
            except Exception as exc:
                logger.error("WS %s: quality metrics computation failed: %s", analysis_id[:8], exc)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for analysis %s — cleaning up (done_count=%d)", analysis_id, done_count)
    except Exception as exc:
        logger.error("WebSocket error for analysis %s: %s (done_count=%d)", analysis_id, exc, done_count)
    finally:
        # Cleanup (Req 8.6) — keep active_results for the REST endpoint
        active_queues.pop(analysis_id, None)
        active_threads.pop(analysis_id, None)
