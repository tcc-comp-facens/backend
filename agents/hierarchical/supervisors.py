"""
Supervisores da arquitetura hierárquica (Nível 1).

Três supervisores especializados coordenam os 8 agentes de nível 2:

- **SupervisorDominio** coordena 4 agentes de domínio (Req 10.2).
- **SupervisorAnalitico** coordena 3 agentes analíticos (Req 10.3).
- **SupervisorContexto** coordena AgenteContextoOrcamentario (Req 10.4).

Todos implementam ``receive_from_peer`` para comunicação lateral
direta entre supervisores do mesmo nível, sem intermediação do
CoordenadorGeral (Reqs 10.5, 10.6).

Requisitos: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7
"""

from __future__ import annotations

import logging
import uuid
from queue import Queue
from typing import Any, TYPE_CHECKING

from agents.base import AgenteBDI
from agents.data_crossing import cross_domain_data, detect_data_gaps
from agents.domain.vigilancia_epidemiologica import AgenteVigilanciaEpidemiologica
from agents.domain.saude_hospitalar import AgenteSaudeHospitalar
from agents.domain.atencao_primaria import AgenteAtencaoPrimaria
from agents.domain.mortalidade import AgenteMortalidade
from agents.analytical.correlacao import AgenteCorrelacao
from agents.analytical.anomalias import AgenteAnomalias
from agents.analytical.sintetizador import AgenteSintetizador
from agents.context.contexto_orcamentario import AgenteContextoOrcamentario
from message_counter import MessageCounter
from metrics import MetricsCollector

if TYPE_CHECKING:
    from db.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class SupervisorDominio(AgenteBDI):
    """Supervisor de domínio — nível 1 da topologia hierárquica (Req 10.2).

    Coordena 4 agentes de domínio (nível 2):
    AgenteVigilanciaEpidemiologica, AgenteSaudeHospitalar,
    AgenteAtencaoPrimaria e AgenteMortalidade.

    Executa os 4 agentes em sequência, agrega resultados de despesas
    e indicadores (Req 10.7), e disponibiliza os dados para
    comunicação lateral via ``receive_from_peer`` (Req 10.5).

    Attributes:
        neo4j_client: Cliente Neo4j repassado aos agentes de domínio.
        peer_data: Dados recebidos de supervisores pares via lateral comm.
    """

    def __init__(self, agent_id: str, neo4j_client: "Neo4jClient") -> None:
        super().__init__(agent_id)
        self.neo4j_client = neo4j_client
        self.peer_data: dict[str, Any] = {}

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Percebe parâmetros da análise a partir das crenças."""
        return {
            "analysis_id": self.beliefs.get("analysis_id"),
            "date_from": self.beliefs.get("date_from"),
            "date_to": self.beliefs.get("date_to"),
        }

    def deliberate(self) -> list[dict]:
        """Define desejos: executar agentes de domínio e agregar."""
        desires: list[dict] = []
        if self.beliefs.get("analysis_id"):
            desires.append({"goal": "executar_agentes_dominio"})
            desires.append({"goal": "agregar_resultados"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        return [{"desire": d, "status": "pending"} for d in desires]

    # -- Lateral communication (Req 10.5) --------------------------------

    def receive_from_peer(self, data: dict[str, Any]) -> None:
        """Recebe dados de um supervisor par (comunicação lateral).

        Args:
            data: Dicionário com dados enviados pelo supervisor par.
        """
        self.peer_data.update(data)
        self.update_beliefs({"peer_data": self.peer_data})
        logger.info(
            "SupervisorDominio %s: received peer data with keys %s",
            self.agent_id,
            list(data.keys()),
        )

    # -- Public API called by CoordenadorGeral --------------------------

    def run(
        self,
        analysis_id: str,
        date_from: int | None,
        date_to: int | None,
        counter: MessageCounter,
    ) -> dict[str, Any]:
        """Executa o pipeline de domínio via 4 agentes subordinados.

        1. Instancia 4 agentes de domínio com IDs únicos (Req 10.2).
        2. Executa cada agente em sequência, coletando dados.
        3. Agrega resultados de despesas e indicadores (Req 10.7).
        4. Trata falhas de subordinados graciosamente.
        5. Coleta métricas por subordinado via MetricsCollector.

        Args:
            analysis_id: UUID da análise.
            date_from: Ano inicial do período.
            date_to: Ano final do período.
            counter: MessageCounter compartilhado para contagem de mensagens.

        Returns:
            Dicionário com "despesas" e "indicadores" agregados.
        """
        self.update_beliefs({
            "analysis_id": analysis_id,
            "date_from": date_from,
            "date_to": date_to,
        })

        # -- 1. Instanciar 4 agentes de domínio com IDs únicos --
        vig_id = f"hier-vigilancia-{uuid.uuid4().hex[:8]}"
        hosp_id = f"hier-hospitalar-{uuid.uuid4().hex[:8]}"
        prim_id = f"hier-primaria-{uuid.uuid4().hex[:8]}"
        mort_id = f"hier-mortalidade-{uuid.uuid4().hex[:8]}"

        agente_vigilancia = AgenteVigilanciaEpidemiologica(vig_id, self.neo4j_client)
        agente_hospitalar = AgenteSaudeHospitalar(hosp_id, self.neo4j_client)
        agente_primaria = AgenteAtencaoPrimaria(prim_id, self.neo4j_client)
        agente_mortalidade = AgenteMortalidade(mort_id, self.neo4j_client)

        domain_agents: list[tuple[str, str, Any]] = [
            (vig_id, "vigilancia_epidemiologica", agente_vigilancia),
            (hosp_id, "saude_hospitalar", agente_hospitalar),
            (prim_id, "atencao_primaria", agente_primaria),
            (mort_id, "mortalidade", agente_mortalidade),
        ]

        all_despesas: list[dict] = []
        all_indicadores: list[dict] = []
        self._collectors: list[MetricsCollector] = []

        # -- 2. Executar agentes em sequência --
        for agent_id_str, agent_type, agent in domain_agents:
            mc = MetricsCollector(agent_id_str, agent_type)
            mc.start()
            try:
                result = agent.query(analysis_id, date_from, date_to)
                mc.stop()
                # Req 11.1: 2 messages per call (ida + volta)
                counter.increment(2)
                all_despesas.extend(result.get("despesas", []))
                all_indicadores.extend(result.get("indicadores", []))
                logger.info(
                    "SupervisorDominio %s: %s returned %d despesas, %d indicadores",
                    self.agent_id,
                    agent_type,
                    len(result.get("despesas", [])),
                    len(result.get("indicadores", [])),
                )
            except Exception as exc:
                mc.stop()
                # Graceful degradation: exclude failed agent, continue
                logger.error(
                    "SupervisorDominio %s: %s failed — %s",
                    self.agent_id,
                    agent_type,
                    exc,
                )
            self._collectors.append(mc)

        # -- 3. Deduplicate despesas (mortalidade returns all subfunções) --
        seen_despesas: set[tuple[int, int]] = set()
        unique_despesas: list[dict] = []
        for d in all_despesas:
            key = (d.get("subfuncao", 0), d.get("ano", 0))
            if key not in seen_despesas:
                seen_despesas.add(key)
                unique_despesas.append(d)

        aggregated = {
            "despesas": unique_despesas,
            "indicadores": all_indicadores,
        }

        self.beliefs["aggregated"] = aggregated
        logger.info(
            "SupervisorDominio %s: aggregated %d despesas, %d indicadores",
            self.agent_id,
            len(unique_despesas),
            len(all_indicadores),
        )

        return aggregated


class SupervisorAnalitico(AgenteBDI):
    """Supervisor analítico — nível 1 da topologia hierárquica (Req 10.3).

    Coordena 3 agentes analíticos (nível 2): AgenteCorrelacao,
    AgenteAnomalias e AgenteSintetizador.

    Recebe dados de domínio e contexto orçamentário via
    ``receive_from_peer`` (Reqs 10.5, 10.6), cruza dados usando
    ``cross_domain_data()``, e executa o pipeline analítico:
    correlação → anomalias → sintetizador.

    Attributes:
        peer_data: Dados recebidos de supervisores pares via lateral comm.
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.peer_data: dict[str, Any] = {}

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Percebe dados disponíveis (recebidos dos peers ou crenças)."""
        return {
            "analysis_id": self.beliefs.get("analysis_id"),
            "despesas": self.peer_data.get("despesas", []),
            "indicadores": self.peer_data.get("indicadores", []),
            "contexto_orcamentario": self.peer_data.get("contexto_orcamentario", {}),
            "ws_queue": self.beliefs.get("ws_queue"),
        }

    def deliberate(self) -> list[dict]:
        """Define desejos: executar pipeline analítico."""
        desires: list[dict] = []
        if self.beliefs.get("analysis_id") and self.beliefs.get("ws_queue") is not None:
            desires.append({"goal": "executar_pipeline_analitico"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        return [{"desire": d, "status": "pending"} for d in desires]

    # -- Lateral communication (Reqs 10.5, 10.6) ------------------------

    def receive_from_peer(self, data: dict[str, Any]) -> None:
        """Recebe dados de um supervisor par (comunicação lateral).

        Tipicamente chamado pelo CoordenadorGeral para repassar dados
        do SupervisorDominio (despesas, indicadores) e do
        SupervisorContexto (contexto_orcamentario).

        Args:
            data: Dicionário com dados enviados pelo supervisor par.
        """
        self.peer_data.update(data)
        self.update_beliefs({"peer_data": self.peer_data})
        logger.info(
            "SupervisorAnalitico %s: received peer data with keys %s",
            self.agent_id,
            list(data.keys()),
        )

    # -- Public API called by CoordenadorGeral --------------------------

    def run(
        self,
        analysis_id: str,
        ws_queue: Queue,
        counter: MessageCounter,
    ) -> dict[str, Any]:
        """Executa o pipeline analítico via 3 agentes subordinados.

        Espera que ``receive_from_peer`` já tenha sido chamado com
        despesas, indicadores e contexto_orcamentario.

        Pipeline:
        1. Cruza dados de domínio usando ``cross_domain_data()``.
        2. AgenteCorrelacao.compute(dados_cruzados).
        3. AgenteAnomalias.detect(dados_cruzados).
        4. AgenteSintetizador.synthesize(correlacoes, anomalias, contexto, ...).

        Args:
            analysis_id: UUID da análise em andamento.
            ws_queue: Fila para streaming de eventos WebSocket.
            counter: MessageCounter compartilhado para contagem de mensagens.

        Returns:
            Dicionário com "correlacoes", "anomalias" e "texto_analise".
        """
        self.update_beliefs({
            "analysis_id": analysis_id,
            "ws_queue": ws_queue,
        })

        # -- 1. Instanciar 3 agentes analíticos com IDs únicos --
        corr_id = f"hier-correlacao-{uuid.uuid4().hex[:8]}"
        anom_id = f"hier-anomalias-{uuid.uuid4().hex[:8]}"
        sint_id = f"hier-sintetizador-{uuid.uuid4().hex[:8]}"

        agente_correlacao = AgenteCorrelacao(corr_id)
        agente_anomalias = AgenteAnomalias(anom_id)
        agente_sintetizador = AgenteSintetizador(sint_id)

        self._collectors: list[MetricsCollector] = []

        # -- 2. Cross domain data (Req 10.5) --
        despesas = self.peer_data.get("despesas", [])
        indicadores = self.peer_data.get("indicadores", [])
        contexto_orcamentario = self.peer_data.get("contexto_orcamentario", {})

        dados_cruzados = cross_domain_data(despesas, indicadores)

        # Detect data gaps for transparency
        date_from = self.peer_data.get("date_from")
        date_to = self.peer_data.get("date_to")
        data_coverage: dict = {}
        if date_from is not None and date_to is not None:
            data_coverage = detect_data_gaps(
                despesas, indicadores, date_from, date_to
            )
            if data_coverage.get("summary", {}).get("has_gaps"):
                logger.warning(
                    "SupervisorAnalitico %s: %d data gaps detected",
                    self.agent_id,
                    data_coverage["summary"]["total_gaps"],
                )

        # -- 3. Correlação --
        correlacoes: list[dict] = []
        mc_corr = MetricsCollector(corr_id, "correlacao")
        mc_corr.start()
        try:
            correlacoes = agente_correlacao.compute(dados_cruzados)
            mc_corr.stop()
            counter.increment(2)
            logger.info(
                "SupervisorAnalitico %s: computed %d correlações",
                self.agent_id,
                len(correlacoes),
            )
        except Exception as exc:
            mc_corr.stop()
            logger.error(
                "SupervisorAnalitico %s: correlacao failed — %s",
                self.agent_id,
                exc,
            )
        self._collectors.append(mc_corr)

        # -- 4. Anomalias --
        anomalias: list[dict] = []
        mc_anom = MetricsCollector(anom_id, "anomalias")
        mc_anom.start()
        try:
            anomalias = agente_anomalias.detect(dados_cruzados)
            mc_anom.stop()
            counter.increment(2)
            logger.info(
                "SupervisorAnalitico %s: detected %d anomalias",
                self.agent_id,
                len(anomalias),
            )
        except Exception as exc:
            mc_anom.stop()
            logger.error(
                "SupervisorAnalitico %s: anomalias failed — %s",
                self.agent_id,
                exc,
            )
        self._collectors.append(mc_anom)

        # -- 5. Sintetizador --
        texto_analise: str = ""
        mc_sint = MetricsCollector(sint_id, "sintetizador")
        mc_sint.start()
        try:
            texto_analise = agente_sintetizador.synthesize(
                correlacoes=correlacoes,
                anomalias=anomalias,
                contexto_orcamentario=contexto_orcamentario,
                analysis_id=analysis_id,
                ws_queue=ws_queue,
                architecture="hierarchical",
                data_coverage=data_coverage,
            )
            mc_sint.stop()
            counter.increment(2)
            logger.info(
                "SupervisorAnalitico %s: synthesis complete (%d chars)",
                self.agent_id,
                len(texto_analise),
            )
        except Exception as exc:
            mc_sint.stop()
            logger.error(
                "SupervisorAnalitico %s: sintetizador failed — %s",
                self.agent_id,
                exc,
            )
        self._collectors.append(mc_sint)

        result = {
            "correlacoes": correlacoes,
            "anomalias": anomalias,
            "texto_analise": texto_analise,
            "data_coverage": data_coverage,
        }

        self.beliefs["aggregated"] = result
        logger.info(
            "SupervisorAnalitico %s: pipeline complete — %d correlacoes, %d anomalias",
            self.agent_id,
            len(correlacoes),
            len(anomalias),
        )

        return result


class SupervisorContexto(AgenteBDI):
    """Supervisor de contexto — nível 1 da topologia hierárquica (Req 10.4).

    Coordena o AgenteContextoOrcamentario (nível 2) para análise de
    tendências temporais de gasto orçamentário.

    Recebe despesas do SupervisorDominio via ``receive_from_peer``
    (Req 10.6) e executa a análise de tendências.

    Attributes:
        peer_data: Dados recebidos de supervisores pares via lateral comm.
    """

    def __init__(self, agent_id: str) -> None:
        super().__init__(agent_id)
        self.peer_data: dict[str, Any] = {}

    # -- BDI overrides --------------------------------------------------

    def perceive(self) -> dict:
        """Percebe dados disponíveis (recebidos dos peers ou crenças)."""
        return {
            "despesas": self.peer_data.get("despesas", []),
        }

    def deliberate(self) -> list[dict]:
        """Define desejos: executar análise de contexto orçamentário."""
        desires: list[dict] = []
        if self.peer_data.get("despesas"):
            desires.append({"goal": "executar_contexto_orcamentario"})
        self.desires = desires
        return desires

    def plan(self, desires: list[dict]) -> list[dict]:
        return [{"desire": d, "status": "pending"} for d in desires]

    # -- Lateral communication (Req 10.6) --------------------------------

    def receive_from_peer(self, data: dict[str, Any]) -> None:
        """Recebe dados de um supervisor par (comunicação lateral).

        Tipicamente chamado pelo CoordenadorGeral para repassar
        despesas do SupervisorDominio.

        Args:
            data: Dicionário com "despesas" do supervisor par.
        """
        self.peer_data.update(data)
        self.update_beliefs({"peer_data": self.peer_data})
        logger.info(
            "SupervisorContexto %s: received peer data with keys %s",
            self.agent_id,
            list(data.keys()),
        )

    # -- Public API called by CoordenadorGeral --------------------------

    def run(
        self,
        counter: MessageCounter,
    ) -> dict[str, Any]:
        """Executa a análise de contexto orçamentário via subordinado.

        Espera que ``receive_from_peer`` já tenha sido chamado com
        as despesas do SupervisorDominio.

        1. Instancia AgenteContextoOrcamentario com ID único.
        2. Executa analyze_trends() com as despesas recebidas.
        3. Coleta métricas via MetricsCollector.

        Args:
            counter: MessageCounter compartilhado para contagem de mensagens.

        Returns:
            Dicionário com "contexto_orcamentario".
        """
        # -- 1. Instanciar agente de contexto com ID único --
        ctx_id = f"hier-contexto-{uuid.uuid4().hex[:8]}"
        agente_contexto = AgenteContextoOrcamentario(ctx_id)

        self._collectors: list[MetricsCollector] = []

        despesas = self.peer_data.get("despesas", [])

        # -- 2. Executar análise de tendências --
        contexto_orcamentario: dict[int, dict] = {}
        mc_ctx = MetricsCollector(ctx_id, "contexto_orcamentario")
        mc_ctx.start()
        try:
            contexto_orcamentario = agente_contexto.analyze_trends(despesas)
            mc_ctx.stop()
            counter.increment(2)
            logger.info(
                "SupervisorContexto %s: computed trends for %d subfunções",
                self.agent_id,
                len(contexto_orcamentario),
            )
        except Exception as exc:
            mc_ctx.stop()
            logger.error(
                "SupervisorContexto %s: contexto_orcamentario failed — %s",
                self.agent_id,
                exc,
            )
        self._collectors.append(mc_ctx)

        result = {
            "contexto_orcamentario": contexto_orcamentario,
        }

        self.beliefs["aggregated"] = result
        logger.info(
            "SupervisorContexto %s: pipeline complete",
            self.agent_id,
        )

        return result
