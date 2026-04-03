"""Unit tests for AgenteBDI base class."""

import pytest

from agents.base import AgenteBDI, IntentionFailure


class TestAgenteBDIInit:
    """Test agent initialization (Req 7.1, 7.2, 7.3)."""

    def test_creates_with_agent_id(self):
        agent = AgenteBDI("test-agent")
        assert agent.agent_id == "test-agent"

    def test_beliefs_initialized_empty(self):
        agent = AgenteBDI("a1")
        assert agent.beliefs == {}

    def test_desires_initialized_empty(self):
        agent = AgenteBDI("a1")
        assert agent.desires == []

    def test_intentions_initialized_empty(self):
        agent = AgenteBDI("a1")
        assert agent.intentions == []


class TestUpdateBeliefs:
    """Test belief updates (Req 7.4)."""

    def test_updates_beliefs_from_perception(self):
        agent = AgenteBDI("a1")
        agent.update_beliefs({"temperature": 30, "humidity": 80})
        assert agent.beliefs == {"temperature": 30, "humidity": 80}

    def test_merges_with_existing_beliefs(self):
        agent = AgenteBDI("a1")
        agent.beliefs = {"temperature": 25}
        agent.update_beliefs({"humidity": 80})
        assert agent.beliefs == {"temperature": 25, "humidity": 80}

    def test_overwrites_existing_keys(self):
        agent = AgenteBDI("a1")
        agent.beliefs = {"temperature": 25}
        agent.update_beliefs({"temperature": 30})
        assert agent.beliefs["temperature"] == 30


class TestDeliberate:
    """Test deliberation — base returns all desires."""

    def test_returns_all_desires(self):
        agent = AgenteBDI("a1")
        agent.desires = [{"goal": "analyze"}, {"goal": "report"}]
        result = agent.deliberate()
        assert len(result) == 2

    def test_returns_copy_not_reference(self):
        agent = AgenteBDI("a1")
        agent.desires = [{"goal": "analyze"}]
        result = agent.deliberate()
        result.append({"goal": "extra"})
        assert len(agent.desires) == 1


class TestPlan:
    """Test planning — base wraps desires as intentions."""

    def test_creates_intentions_from_desires(self):
        agent = AgenteBDI("a1")
        desires = [{"goal": "analyze"}]
        intentions = agent.plan(desires)
        assert len(intentions) == 1
        assert intentions[0]["desire"] == {"goal": "analyze"}
        assert intentions[0]["status"] == "pending"

    def test_empty_desires_produce_empty_intentions(self):
        agent = AgenteBDI("a1")
        assert agent.plan([]) == []


class TestExecute:
    """Test execution with failure recovery (Req 7.5)."""

    def test_executes_intentions_successfully(self):
        agent = AgenteBDI("a1")
        intentions = [{"desire": {"goal": "x"}, "status": "pending"}]
        agent.execute(intentions)
        assert intentions[0]["status"] == "completed"

    def test_handles_intention_failure(self):
        """When an intention fails, status is 'failed', not undefined."""

        class FailingAgent(AgenteBDI):
            def _execute_intention(self, intention):
                raise IntentionFailure(intention, "test failure")

        agent = FailingAgent("a1")
        intentions = [{"desire": {"goal": "x"}, "status": "pending"}]
        agent.execute(intentions)
        assert intentions[0]["status"] == "failed"

    def test_tries_alternative_on_failure(self):
        """Req 7.5: selects alternative intention when one fails."""

        class RecoverableAgent(AgenteBDI):
            def __init__(self, agent_id):
                super().__init__(agent_id)
                self._call_count = 0

            def _execute_intention(self, intention):
                self._call_count += 1
                if self._call_count == 1:
                    raise IntentionFailure(intention, "first attempt failed")
                intention["status"] = "completed"

            def _recover_intention(self, failed):
                return {"desire": failed["desire"], "status": "pending", "alternative": True}

        agent = RecoverableAgent("a1")
        intentions = [{"desire": {"goal": "x"}, "status": "pending"}]
        agent.execute(intentions)
        # Original failed, but alternative was tried
        assert intentions[0]["status"] == "failed"
        assert agent._call_count == 2

    def test_no_undefined_state_after_failure(self):
        """Req 7.5: agent never stays in undefined state."""

        class FailingAgent(AgenteBDI):
            def _execute_intention(self, intention):
                raise IntentionFailure(intention, "always fails")

        agent = FailingAgent("a1")
        intentions = [
            {"desire": {"goal": "a"}, "status": "pending"},
            {"desire": {"goal": "b"}, "status": "pending"},
        ]
        agent.execute(intentions)
        for i in intentions:
            assert i["status"] == "failed"


class TestRunCycle:
    """Test full BDI cycle."""

    def test_full_cycle_updates_all_attributes(self):
        class TestAgent(AgenteBDI):
            def perceive(self):
                return {"data_available": True}

            def deliberate(self):
                if self.beliefs.get("data_available"):
                    return [{"goal": "analyze"}]
                return []

        agent = TestAgent("a1")
        agent.desires = [{"goal": "analyze"}]
        agent.run_cycle()

        assert agent.beliefs.get("data_available") is True
        assert len(agent.intentions) > 0

    def test_cycle_follows_bdi_order(self):
        """Verify perceive → update_beliefs → deliberate → plan → execute order."""
        call_order = []

        class OrderAgent(AgenteBDI):
            def perceive(self):
                call_order.append("perceive")
                return {"step": 1}

            def update_beliefs(self, perception):
                call_order.append("update_beliefs")
                super().update_beliefs(perception)

            def deliberate(self):
                call_order.append("deliberate")
                return super().deliberate()

            def plan(self, desires):
                call_order.append("plan")
                return super().plan(desires)

            def execute(self, intentions):
                call_order.append("execute")
                super().execute(intentions)

        agent = OrderAgent("a1")
        agent.run_cycle()
        assert call_order == ["perceive", "update_beliefs", "deliberate", "plan", "execute"]


# Feature: multiagent-architecture-comparison, Property 10: Ciclo BDI completo
# Validates: Requirements 7.1, 7.2, 7.3

from hypothesis import given, settings
from hypothesis import strategies as st


class TestP10CicloBDICompleto:
    """Property test P10: After a full BDI cycle with non-empty perception,
    beliefs, desires, and intentions must all be defined and non-empty.

    **Validates: Requirements 7.1, 7.2, 7.3**
    """

    # Strategy: generate non-empty dicts with text keys and mixed values
    perception_strategy = st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(min_size=1, max_size=50),
            st.integers(min_value=-1000, max_value=1000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
        min_size=1,
        max_size=10,
    )

    @given(perception=perception_strategy)
    @settings(max_examples=100)
    def test_p10_full_bdi_cycle_sets_beliefs_desires_intentions(self, perception):
        """For any non-empty perception, after run_cycle the agent must have
        non-empty beliefs, desires (from deliberate), and intentions."""

        class ConcreteBDIAgent(AgenteBDI):
            """Concrete subclass that uses generated perception and
            produces desires from beliefs so the cycle is fully exercised."""

            def __init__(self, agent_id: str, injected_perception: dict):
                super().__init__(agent_id)
                self._injected_perception = injected_perception

            def perceive(self) -> dict:
                return self._injected_perception

            def deliberate(self) -> list[dict]:
                # Produce one desire per belief key — guarantees non-empty
                # when beliefs are non-empty
                return [{"goal": k} for k in self.beliefs]

        agent = ConcreteBDIAgent("test-p10", perception)
        agent.run_cycle()

        # Req 7.1: beliefs must be defined and non-empty
        assert isinstance(agent.beliefs, dict)
        assert len(agent.beliefs) > 0, "beliefs should not be empty after cycle"

        # Req 7.2: desires produced by deliberate must be non-empty
        # (deliberate is called inside run_cycle; we verify via intentions
        #  which are derived from desires through plan)
        assert isinstance(agent.intentions, list)
        assert len(agent.intentions) > 0, "intentions should not be empty after cycle"

        # Req 7.3: each intention wraps a desire and has a status
        for intention in agent.intentions:
            assert "desire" in intention, "intention must reference a desire"
            assert "status" in intention, "intention must have a status"
            assert intention["status"] in (
                "pending",
                "completed",
                "failed",
            ), f"unexpected status: {intention['status']}"

# Feature: multiagent-architecture-comparison, Property 11: Atualização de crenças
# Validates: Requirement 7.4


class TestP11AtualizacaoDeCrencas:
    """Property test P11: After update_beliefs(perception), the agent's belief
    base must reflect ALL keys from the perception received.

    **Validates: Requirement 7.4**
    """

    # Strategy: generate non-empty perception dicts with varied value types
    perception_strategy = st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=50),
            st.integers(min_value=-10000, max_value=10000),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
            st.none(),
        ),
        min_size=1,
        max_size=15,
    )

    # Strategy: generate optional pre-existing beliefs
    existing_beliefs_strategy = st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.one_of(
            st.text(max_size=50),
            st.integers(min_value=-10000, max_value=10000),
            st.booleans(),
        ),
        min_size=0,
        max_size=10,
    )

    @given(perception=perception_strategy)
    @settings(max_examples=100)
    def test_p11_all_perception_keys_present_in_beliefs(self, perception):
        """After update_beliefs, every key from the perception must exist
        in the agent's beliefs with the correct value."""
        agent = AgenteBDI("test-p11")
        agent.update_beliefs(perception)

        for key, value in perception.items():
            assert key in agent.beliefs, f"key '{key}' missing from beliefs"
            assert agent.beliefs[key] == value, (
                f"beliefs['{key}'] = {agent.beliefs[key]!r}, expected {value!r}"
            )

    @given(
        existing=existing_beliefs_strategy,
        perception=perception_strategy,
    )
    @settings(max_examples=100)
    def test_p11_perception_merges_with_existing_beliefs(self, existing, perception):
        """Pre-existing beliefs that are NOT overwritten by the perception
        must be preserved after update_beliefs."""
        agent = AgenteBDI("test-p11-merge")
        agent.beliefs = dict(existing)
        agent.update_beliefs(perception)

        # All perception keys must be reflected
        for key, value in perception.items():
            assert agent.beliefs[key] == value

        # Existing keys not in perception must survive
        for key, value in existing.items():
            if key not in perception:
                assert key in agent.beliefs, (
                    f"existing key '{key}' was lost after update_beliefs"
                )
                assert agent.beliefs[key] == value

    @given(perception=perception_strategy)
    @settings(max_examples=100)
    def test_p11_beliefs_contain_no_extra_keys(self, perception):
        """Starting from empty beliefs, after update_beliefs the belief base
        must contain exactly the keys from the perception — no more, no less."""
        agent = AgenteBDI("test-p11-exact")
        agent.update_beliefs(perception)

        assert set(agent.beliefs.keys()) == set(perception.keys()), (
            f"beliefs keys {set(agent.beliefs.keys())} != perception keys {set(perception.keys())}"
        )



# Feature: multiagent-architecture-comparison, Property 12: Recuperação de intenção
# Validates: Requirement 7.5


class TestP12RecuperacaoDeIntencao:
    """Property test P12: When an intention fails, the agent must select an
    alternative intention (if available) or report failure — never remain in
    an undefined state.

    **Validates: Requirements 7.5**
    """

    # Strategy: generate a list of intentions with varied goals
    intentions_strategy = st.lists(
        st.dictionaries(
            keys=st.just("goal"),
            values=st.text(min_size=1, max_size=30),
            min_size=1,
            max_size=1,
        ),
        min_size=1,
        max_size=8,
    )

    # Strategy: which indices should fail (as a set of booleans per intention)
    failure_flags_strategy = st.lists(
        st.booleans(),
        min_size=1,
        max_size=8,
    )

    @given(
        desires=intentions_strategy,
        failure_flags=failure_flags_strategy,
    )
    @settings(max_examples=100)
    def test_p12_no_undefined_status_after_all_failures(self, desires, failure_flags):
        """When ALL intentions fail and no recovery is available, every
        intention must have status 'failed' — never 'pending' or undefined."""

        class AlwaysFailAgent(AgenteBDI):
            def _execute_intention(self, intention):
                raise IntentionFailure(intention, "simulated failure")

        agent = AlwaysFailAgent("test-p12-all-fail")
        intentions = [{"desire": d, "status": "pending"} for d in desires]
        agent.execute(intentions)

        for intention in intentions:
            assert "status" in intention, "intention must have a status field"
            assert intention["status"] in ("completed", "failed"), (
                f"status must be 'completed' or 'failed', got '{intention['status']}'"
            )

    @given(desires=intentions_strategy)
    @settings(max_examples=100)
    def test_p12_recovery_provides_alternative_and_attempts_it(self, desires):
        """When an intention fails and recovery provides an alternative,
        the agent must attempt the alternative. The alternative ends up
        either 'completed' or 'failed' — never undefined."""

        class RecoverOnceAgent(AgenteBDI):
            def __init__(self, agent_id):
                super().__init__(agent_id)
                self._original_ids: set[int] = set()
                self._alternatives_attempted: int = 0

            def _execute_intention(self, intention):
                obj_id = id(intention)
                if obj_id in self._original_ids:
                    # This is an original intention — always fail
                    raise IntentionFailure(intention, "original always fails")
                # This is an alternative — succeed
                self._alternatives_attempted += 1
                intention["status"] = "completed"

            def _recover_intention(self, failed):
                return {
                    "desire": failed["desire"],
                    "status": "pending",
                    "alternative": True,
                }

        agent = RecoverOnceAgent("test-p12-recover")
        intentions = [{"desire": d, "status": "pending"} for d in desires]
        # Register original intention ids so we can distinguish them
        for i in intentions:
            agent._original_ids.add(id(i))
        agent.execute(intentions)

        # Original intentions are marked failed
        for intention in intentions:
            assert intention["status"] == "failed", (
                f"original intention should be 'failed', got '{intention['status']}'"
            )

        # The agent attempted recovery for each intention
        assert agent._alternatives_attempted == len(desires), (
            f"expected {len(desires)} alternatives attempted, got {agent._alternatives_attempted}"
        )

    @given(
        desires=intentions_strategy,
        failure_flags=failure_flags_strategy,
    )
    @settings(max_examples=100)
    def test_p12_mixed_success_failure_no_undefined(self, desires, failure_flags):
        """For any mix of succeeding and failing intentions, every intention
        must end with a definite status ('completed' or 'failed')."""

        # Align failure_flags length with desires
        flags = failure_flags[: len(desires)]
        while len(flags) < len(desires):
            flags.append(False)

        class MixedAgent(AgenteBDI):
            def __init__(self, agent_id, should_fail):
                super().__init__(agent_id)
                self._should_fail = should_fail
                self._index = 0

            def _execute_intention(self, intention):
                idx = self._index
                self._index += 1
                if idx < len(self._should_fail) and self._should_fail[idx]:
                    raise IntentionFailure(intention, f"intention {idx} fails")
                intention["status"] = "completed"

        agent = MixedAgent("test-p12-mixed", flags)
        intentions = [{"desire": d, "status": "pending"} for d in desires]
        agent.execute(intentions)

        for i, intention in enumerate(intentions):
            assert "status" in intention, f"intention {i} missing status"
            assert intention["status"] in ("completed", "failed"), (
                f"intention {i} has undefined status '{intention['status']}'"
            )

    @given(desires=intentions_strategy)
    @settings(max_examples=100)
    def test_p12_failed_intentions_tracked(self, desires):
        """When intentions fail, they must be recorded in _failed_intentions,
        confirming the agent reports failure rather than silently ignoring it."""

        class AlwaysFailAgent(AgenteBDI):
            def _execute_intention(self, intention):
                raise IntentionFailure(intention, "always fails")

        agent = AlwaysFailAgent("test-p12-tracked")
        intentions = [{"desire": d, "status": "pending"} for d in desires]
        agent.execute(intentions)

        assert len(agent._failed_intentions) == len(desires), (
            f"expected {len(desires)} failed intentions, got {len(agent._failed_intentions)}"
        )
