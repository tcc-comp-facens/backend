# Feature: multiagent-architecture-comparison, Property 19: Isolamento entre arquiteturas
"""
Property-based tests for architecture isolation.

**Validates: Requirements 5.1, 6.1**

Verifies that agent instances from the star and hierarchical architectures
do not share mutable state. Each architecture operates on its own agent
instances so that concurrent execution (one thread per architecture) cannot
cause cross-contamination of beliefs, desires, or intentions.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
from queue import Queue
from unittest.mock import MagicMock, patch

from hypothesis import given, settings
from hypothesis import strategies as st

from agents.base import AgenteBDI
from agents.star.consultant import AgenteConsultorStar
from agents.star.analyzer import AgenteAnalisadorStar
from agents.star.orchestrator import OrquestradorEstrela

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

VALID_SUBFUNCOES = [301, 302, 303, 305]
SUBFUNCAO_NOMES = {
    301: "Atenção Básica",
    302: "Assistência Hospitalar",
    303: "Suporte Profilático",
    305: "Vigilância Epidemiológica",
}
VALID_HEALTH_PARAMS = ["dengue", "covid", "vacinacao", "mortalidade", "internacoes"]


@st.composite
def beliefs_strategy(draw):
    """Generate arbitrary beliefs dictionaries."""
    keys = draw(
        st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("L", "N")),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
            unique=True,
        )
    )
    values = draw(
        st.lists(
            st.one_of(
                st.integers(min_value=-1000, max_value=1000),
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                st.text(min_size=0, max_size=20),
                st.lists(st.integers(min_value=0, max_value=100), max_size=5),
            ),
            min_size=len(keys),
            max_size=len(keys),
        )
    )
    return dict(zip(keys, values))


@st.composite
def desires_strategy(draw):
    """Generate a list of desire dicts."""
    return draw(
        st.lists(
            st.fixed_dictionaries({"goal": st.text(min_size=1, max_size=15)}),
            min_size=0,
            max_size=4,
        )
    )


@st.composite
def agent_id_pair(draw):
    """Generate two distinct agent IDs."""
    id1 = draw(st.text(min_size=3, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-"))
    id2 = draw(st.text(min_size=3, max_size=15, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-").filter(lambda x: x != id1))
    return id1, id2


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    beliefs_star=beliefs_strategy(),
    beliefs_hier=beliefs_strategy(),
    desires_star=desires_strategy(),
    desires_hier=desires_strategy(),
    ids=agent_id_pair(),
)
def test_p19_base_agents_no_shared_mutable_state(
    beliefs_star, beliefs_hier, desires_star, desires_hier, ids
):
    """
    Property 19: Isolamento entre arquiteturas — base AgenteBDI.

    For any two AgenteBDI instances (one representing star, one hierarchical),
    mutating the beliefs, desires, or intentions of one must never affect the other.
    """
    star_id, hier_id = ids

    agent_star = AgenteBDI(f"star-{star_id}")
    agent_hier = AgenteBDI(f"hier-{hier_id}")

    # Update beliefs independently
    agent_star.update_beliefs(beliefs_star)
    agent_hier.update_beliefs(beliefs_hier)

    # Verify beliefs are independent
    assert agent_star.beliefs is not agent_hier.beliefs, (
        "Star and hierarchical agents must not share the same beliefs dict"
    )
    for key in beliefs_star:
        if key in beliefs_hier:
            # Values may coincide but the containers must be distinct objects
            pass
        assert key in agent_star.beliefs

    for key in beliefs_hier:
        assert key in agent_hier.beliefs

    # Mutate star beliefs — hierarchical must be unaffected
    snapshot_hier = copy.deepcopy(agent_hier.beliefs)
    agent_star.update_beliefs({"__isolation_probe__": True})
    assert "__isolation_probe__" not in agent_hier.beliefs, (
        "Mutating star agent beliefs must not affect hierarchical agent"
    )
    assert agent_hier.beliefs == snapshot_hier

    # Set desires independently
    agent_star.desires = list(desires_star)
    agent_hier.desires = list(desires_hier)

    assert agent_star.desires is not agent_hier.desires, (
        "Star and hierarchical agents must not share the same desires list"
    )

    # Mutate star desires — hierarchical must be unaffected
    snapshot_hier_desires = list(agent_hier.desires)
    agent_star.desires.append({"goal": "__probe__"})
    assert {"goal": "__probe__"} not in agent_hier.desires, (
        "Mutating star agent desires must not affect hierarchical agent"
    )
    assert agent_hier.desires == snapshot_hier_desires

    # Intentions isolation
    agent_star.intentions = [{"desire": d, "status": "pending"} for d in desires_star]
    agent_hier.intentions = [{"desire": d, "status": "pending"} for d in desires_hier]

    assert agent_star.intentions is not agent_hier.intentions
    snapshot_hier_intentions = copy.deepcopy(agent_hier.intentions)
    agent_star.intentions.append({"desire": {"goal": "__probe__"}, "status": "pending"})
    assert len(agent_hier.intentions) == len(snapshot_hier_intentions), (
        "Mutating star agent intentions must not affect hierarchical agent"
    )


@settings(max_examples=100)
@given(
    ids=agent_id_pair(),
    beliefs1=beliefs_strategy(),
    beliefs2=beliefs_strategy(),
)
def test_p19_consultant_instances_isolated(ids, beliefs1, beliefs2):
    """
    Property 19: Isolamento entre arquiteturas — AgenteConsultorStar instances.

    Two consultant instances (simulating star vs hierarchical) must have
    completely independent state.
    """
    id1, id2 = ids
    mock_neo4j_1 = MagicMock()
    mock_neo4j_2 = MagicMock()

    consul_star = AgenteConsultorStar(f"star-consul-{id1}", mock_neo4j_1)
    consul_hier = AgenteConsultorStar(f"hier-consul-{id2}", mock_neo4j_2)

    consul_star.update_beliefs(beliefs1)
    consul_hier.update_beliefs(beliefs2)

    # Beliefs must be distinct objects
    assert consul_star.beliefs is not consul_hier.beliefs
    assert consul_star.neo4j_client is not consul_hier.neo4j_client

    # Mutation isolation
    consul_star.update_beliefs({"__probe__": 999})
    assert "__probe__" not in consul_hier.beliefs


@settings(max_examples=100)
@given(
    ids=agent_id_pair(),
    beliefs1=beliefs_strategy(),
    beliefs2=beliefs_strategy(),
)
def test_p19_analyzer_instances_isolated(ids, beliefs1, beliefs2):
    """
    Property 19: Isolamento entre arquiteturas — AgenteAnalisadorStar instances.

    Two analyzer instances must have completely independent state.
    """
    id1, id2 = ids

    analyzer_star = AgenteAnalisadorStar(f"star-anal-{id1}")
    analyzer_hier = AgenteAnalisadorStar(f"hier-anal-{id2}")

    analyzer_star.update_beliefs(beliefs1)
    analyzer_hier.update_beliefs(beliefs2)

    assert analyzer_star.beliefs is not analyzer_hier.beliefs

    analyzer_star.update_beliefs({"__probe__": "star_only"})
    assert "__probe__" not in analyzer_hier.beliefs


@settings(max_examples=100)
@given(ids=agent_id_pair())
def test_p19_orchestrator_instances_isolated(ids):
    """
    Property 19: Isolamento entre arquiteturas — OrquestradorEstrela instances.

    Two orchestrator instances running in parallel threads must not share
    any mutable state. Each creates its own peripheral agents internally.
    """
    id1, id2 = ids
    mock_neo4j_1 = MagicMock()
    mock_neo4j_2 = MagicMock()

    orch1 = OrquestradorEstrela(f"star-orch-{id1}", mock_neo4j_1)
    orch2 = OrquestradorEstrela(f"hier-orch-{id2}", mock_neo4j_2)

    # Core mutable attributes must be distinct objects
    assert orch1.beliefs is not orch2.beliefs
    assert orch1.desires is not orch2.desires
    assert orch1.intentions is not orch2.intentions
    assert orch1.neo4j_client is not orch2.neo4j_client

    # Mutating one must not affect the other
    orch1.update_beliefs({"analysis_id": "aaa", "status": "running"})
    orch2.update_beliefs({"analysis_id": "bbb", "status": "pending"})

    assert orch1.beliefs["analysis_id"] == "aaa"
    assert orch2.beliefs["analysis_id"] == "bbb"
    assert "status" in orch1.beliefs and orch1.beliefs["status"] == "running"
    assert "status" in orch2.beliefs and orch2.beliefs["status"] == "pending"


@settings(max_examples=100)
@given(ids=agent_id_pair())
def test_p19_parallel_run_no_shared_state(ids):
    """
    Property 19: Isolamento entre arquiteturas — full run isolation.

    Simulates two orchestrator runs (star and hierarchical threads) and
    verifies that the internal agents created by each run do not share
    mutable state. The ws_queues, consultant data, and analyzer results
    must be completely independent.
    """
    id1, id2 = ids
    mock_neo4j_1 = MagicMock()
    mock_neo4j_2 = MagicMock()

    orch_star = OrquestradorEstrela(f"star-{id1}", mock_neo4j_1)
    orch_hier = OrquestradorEstrela(f"hier-{id2}", mock_neo4j_2)

    ws_queue_star = Queue()
    ws_queue_hier = Queue()

    # Queues must be distinct
    assert ws_queue_star is not ws_queue_hier

    consultant_data_star = {
        "despesas": [{"subfuncao": 301, "ano": 2020, "valor": 100.0}],
        "indicadores": [{"tipo": "vacinacao", "ano": 2020, "valor": 80.0}],
    }
    consultant_data_hier = {
        "despesas": [{"subfuncao": 302, "ano": 2021, "valor": 200.0}],
        "indicadores": [{"tipo": "internacoes", "ano": 2021, "valor": 50.0}],
    }

    # Run star orchestrator with mocked agents
    with patch("agents.star.orchestrator.AgenteConsultorStar") as MockCons1, \
         patch("agents.star.orchestrator.AgenteAnalisadorStar") as MockAnal1:
        MockCons1.return_value.query.return_value = consultant_data_star
        MockAnal1.return_value.analyze.return_value = {
            "correlacoes": [], "anomalias": [], "texto_analise": "Star analysis"
        }
        orch_star.run("analysis-star", {"date_from": 2020, "date_to": 2020, "health_params": ["vacinacao"]}, ws_queue_star)

    # Run hierarchical orchestrator with mocked agents
    with patch("agents.star.orchestrator.AgenteConsultorStar") as MockCons2, \
         patch("agents.star.orchestrator.AgenteAnalisadorStar") as MockAnal2:
        MockCons2.return_value.query.return_value = consultant_data_hier
        MockAnal2.return_value.analyze.return_value = {
            "correlacoes": [], "anomalias": [], "texto_analise": "Hier analysis"
        }
        orch_hier.run("analysis-hier", {"date_from": 2021, "date_to": 2021, "health_params": ["internacoes"]}, ws_queue_hier)

    # Verify beliefs are completely independent after runs
    assert orch_star.beliefs["analysis_id"] == "analysis-star"
    assert orch_hier.beliefs["analysis_id"] == "analysis-hier"
    assert orch_star.beliefs is not orch_hier.beliefs

    # Verify result isolation
    assert orch_star.beliefs.get("result", {}).get("texto_analise") == "Star analysis"
    assert orch_hier.beliefs.get("result", {}).get("texto_analise") == "Hier analysis"

    # Verify queues received independent events (no cross-contamination)
    star_events = []
    while not ws_queue_star.empty():
        star_events.append(ws_queue_star.get_nowait())

    hier_events = []
    while not ws_queue_hier.empty():
        hier_events.append(ws_queue_hier.get_nowait())

    # No star events should appear in hier queue and vice versa
    for evt in star_events:
        assert evt.get("analysisId") != "analysis-hier", (
            "Star queue must not contain hierarchical events"
        )
    for evt in hier_events:
        assert evt.get("analysisId") != "analysis-star", (
            "Hierarchical queue must not contain star events"
        )
