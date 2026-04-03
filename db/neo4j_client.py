"""
Cliente Neo4j para o sistema de comparação de arquiteturas multiagente.

Gerencia conexão com o banco de dados Neo4j e expõe queries Cypher
para os nós DespesaSIOPS, IndicadorDataSUS, Analise e MetricaExecucao.
"""

import os
import json
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


class Neo4jClient:
    """Driver Neo4j com queries Cypher para o domínio de análise de saúde."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self._uri = uri or os.environ["NEO4J_URI"]
        self._user = user or os.environ["NEO4J_USER"]
        self._password = password or os.environ["NEO4J_PASSWORD"]
        self._driver = GraphDatabase.driver(
            self._uri, auth=(self._user, self._password)
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Fecha a conexão com o Neo4j."""
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Consultas de leitura
    # ------------------------------------------------------------------

    def get_despesas(
        self, analysis_id: str, date_from: int, date_to: int
    ) -> list[dict]:
        """
        Retorna despesas SIOPS vinculadas a uma análise, filtradas por período.

        Requisitos: 3.1, 12.1
        """
        query = """
        MATCH (a:Analise {id: $analysisId})-[:POSSUI_DESPESA]->(d:DespesaSIOPS)
        WHERE d.ano >= $dateFrom AND d.ano <= $dateTo
        RETURN d.subfuncao AS subfuncao, d.subfuncaoNome AS subfuncaoNome,
               d.ano AS ano, d.valor AS valor
        ORDER BY d.ano, d.subfuncao
        """
        with self._driver.session() as session:
            result = session.run(
                query,
                analysisId=analysis_id,
                dateFrom=date_from,
                dateTo=date_to,
            )
            return [dict(record) for record in result]

    def get_indicadores(
        self,
        analysis_id: str,
        date_from: int,
        date_to: int,
        health_params: list[str],
    ) -> list[dict]:
        """
        Retorna indicadores DataSUS vinculados a uma análise, filtrados por
        período e tipos de indicador.

        Requisitos: 3.1, 3.3, 12.2
        """
        query = """
        MATCH (a:Analise {id: $analysisId})-[:POSSUI_INDICADOR]->(i:IndicadorDataSUS)
        WHERE i.ano >= $dateFrom AND i.ano <= $dateTo
          AND i.tipo IN $healthParams
        RETURN i.tipo AS tipo, i.ano AS ano, i.valor AS valor
        ORDER BY i.ano, i.tipo
        """
        with self._driver.session() as session:
            result = session.run(
                query,
                analysisId=analysis_id,
                dateFrom=date_from,
                dateTo=date_to,
                healthParams=health_params,
            )
            return [dict(record) for record in result]

    def get_correlacoes(self, date_from: int, date_to: int) -> list[dict]:
        """
        Retorna correlações entre despesas e indicadores para um período.

        Requisitos: 4.1, 12.1, 12.2
        """
        query = """
        MATCH (d:DespesaSIOPS)-[:CORRELACIONA_COM]->(i:IndicadorDataSUS)
        WHERE d.ano >= $dateFrom AND d.ano <= $dateTo
          AND d.ano = i.ano
        RETURN d.subfuncao AS subfuncao, d.valor AS despesa,
               i.tipo AS tipo, i.valor AS indicador, d.ano AS ano
        ORDER BY d.ano
        """
        with self._driver.session() as session:
            result = session.run(
                query,
                dateFrom=date_from,
                dateTo=date_to,
            )
            return [dict(record) for record in result]

    def get_benchmarks(self, analysis_id: str) -> list[dict]:
        """
        Retorna métricas de execução vinculadas a uma análise.

        Requisitos: 11.1, 11.2, 11.3, 12.4
        """
        query = """
        MATCH (a:Analise {id: $analysisId})-[:GEROU_METRICA]->(m:MetricaExecucao)
        RETURN m.architecture AS architecture, m.agentId AS agentId,
               m.executionTimeMs AS executionTimeMs,
               m.cpuPercent AS cpuPercent, m.memoryMb AS memoryMb
        ORDER BY m.architecture, m.agentId
        """
        with self._driver.session() as session:
            result = session.run(query, analysisId=analysis_id)
            return [dict(record) for record in result]

    # ------------------------------------------------------------------
    # Operações de escrita
    # ------------------------------------------------------------------

    def save_analise(self, analise: dict) -> None:
        """
        Persiste (ou atualiza) um nó Analise no Neo4j via MERGE.

        Campos esperados em `analise`:
          id, dateFrom, dateTo, healthParams (dict ou str JSON),
          starStatus, hierStatus, createdAt (opcional)

        Requisitos: 12.3, 12.5
        """
        health_params = analise.get("healthParams", {})
        if isinstance(health_params, dict):
            health_params = json.dumps(health_params)

        created_at = analise.get("createdAt") or datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (a:Analise {id: $id})
        SET a.dateFrom        = $dateFrom,
            a.dateTo          = $dateTo,
            a.healthParams    = $healthParams,
            a.starStatus      = $starStatus,
            a.starTextAnalysis = $starTextAnalysis,
            a.starCompletedAt  = $starCompletedAt,
            a.hierStatus      = $hierStatus,
            a.hierTextAnalysis = $hierTextAnalysis,
            a.hierCompletedAt  = $hierCompletedAt,
            a.createdAt       = $createdAt
        """
        with self._driver.session() as session:
            session.run(
                query,
                id=analise["id"],
                dateFrom=analise.get("dateFrom"),
                dateTo=analise.get("dateTo"),
                healthParams=health_params,
                starStatus=analise.get("starStatus", "pending"),
                starTextAnalysis=analise.get("starTextAnalysis"),
                starCompletedAt=analise.get("starCompletedAt"),
                hierStatus=analise.get("hierStatus", "pending"),
                hierTextAnalysis=analise.get("hierTextAnalysis"),
                hierCompletedAt=analise.get("hierCompletedAt"),
                createdAt=created_at,
            )

    def save_metrica(self, metrica: dict, analysis_id: str) -> None:
        """
        Persiste um nó MetricaExecucao e cria o relacionamento
        (:Analise)-[:GEROU_METRICA]->(:MetricaExecucao).

        Campos esperados em `metrica`:
          id, architecture, agentId, agentType,
          executionTimeMs, cpuPercent, memoryMb, recordedAt (opcional)

        Requisitos: 11.4, 12.4
        """
        recorded_at = metrica.get("recordedAt") or datetime.now(timezone.utc).isoformat()

        query = """
        MERGE (m:MetricaExecucao {id: $id})
        SET m.architecture    = $architecture,
            m.agentId         = $agentId,
            m.agentType       = $agentType,
            m.executionTimeMs = $executionTimeMs,
            m.cpuPercent      = $cpuPercent,
            m.memoryMb        = $memoryMb,
            m.recordedAt      = $recordedAt
        WITH m
        MATCH (a:Analise {id: $analysisId})
        MERGE (a)-[:GEROU_METRICA]->(m)
        """
        with self._driver.session() as session:
            session.run(
                query,
                id=metrica["id"],
                architecture=metrica.get("architecture"),
                agentId=metrica.get("agentId"),
                agentType=metrica.get("agentType"),
                executionTimeMs=metrica.get("executionTimeMs"),
                cpuPercent=metrica.get("cpuPercent"),
                memoryMb=metrica.get("memoryMb"),
                recordedAt=recorded_at,
                analysisId=analysis_id,
            )

    # ------------------------------------------------------------------
    # Helpers de escrita usados pelo ETL
    # ------------------------------------------------------------------

    def save_despesa(self, despesa: dict, analysis_id: Optional[str] = None) -> None:
        """
        Persiste um nó DespesaSIOPS via MERGE e, opcionalmente, vincula a
        uma Analise via POSSUI_DESPESA.

        Requisitos: 12.1
        """
        query = """
        MERGE (d:DespesaSIOPS {id: $id})
        SET d.subfuncao     = $subfuncao,
            d.subfuncaoNome = $subfuncaoNome,
            d.ano           = $ano,
            d.valor         = $valor,
            d.fonte         = $fonte
        """
        params = {
            "id": despesa["id"],
            "subfuncao": despesa.get("subfuncao"),
            "subfuncaoNome": despesa.get("subfuncaoNome"),
            "ano": despesa.get("ano"),
            "valor": despesa.get("valor"),
            "fonte": despesa.get("fonte", "siops"),
        }

        if analysis_id:
            query += """
        WITH d
        MATCH (a:Analise {id: $analysisId})
        MERGE (a)-[:POSSUI_DESPESA]->(d)
            """
            params["analysisId"] = analysis_id

        with self._driver.session() as session:
            session.run(query, **params)

    def save_indicador(
        self, indicador: dict, analysis_id: Optional[str] = None
    ) -> None:
        """
        Persiste um nó IndicadorDataSUS via MERGE e, opcionalmente, vincula
        a uma Analise via POSSUI_INDICADOR.

        Requisitos: 12.2
        """
        query = """
        MERGE (i:IndicadorDataSUS {id: $id})
        SET i.sistema = $sistema,
            i.tipo    = $tipo,
            i.ano     = $ano,
            i.valor   = $valor,
            i.fonte   = $fonte
        """
        params = {
            "id": indicador["id"],
            "sistema": indicador.get("sistema"),
            "tipo": indicador.get("tipo"),
            "ano": indicador.get("ano"),
            "valor": indicador.get("valor"),
            "fonte": indicador.get("fonte", "datasus"),
        }

        if analysis_id:
            query += """
        WITH i
        MATCH (a:Analise {id: $analysisId})
        MERGE (a)-[:POSSUI_INDICADOR]->(i)
            """
            params["analysisId"] = analysis_id

        with self._driver.session() as session:
            session.run(query, **params)
