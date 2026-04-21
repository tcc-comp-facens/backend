"""Unit tests for MessageCounter.

Validates thread-safe atomic counting of inter-agent messages.
Requirements: 11.1, 11.2
"""

from __future__ import annotations

import threading

from message_counter import MessageCounter


class TestMessageCounterBasic:
    """Basic functionality tests."""

    def test_initial_count_is_zero(self) -> None:
        counter = MessageCounter()
        assert counter.count == 0

    def test_default_increment_adds_two(self) -> None:
        counter = MessageCounter()
        counter.increment()
        assert counter.count == 2

    def test_custom_increment_value(self) -> None:
        counter = MessageCounter()
        counter.increment(5)
        assert counter.count == 5

    def test_increment_by_one(self) -> None:
        counter = MessageCounter()
        counter.increment(1)
        assert counter.count == 1

    def test_increment_by_zero(self) -> None:
        counter = MessageCounter()
        counter.increment(0)
        assert counter.count == 0

    def test_multiple_increments_accumulate(self) -> None:
        counter = MessageCounter()
        counter.increment()  # +2
        counter.increment()  # +2
        counter.increment(3)  # +3
        assert counter.count == 7

    def test_negative_increment_raises_value_error(self) -> None:
        counter = MessageCounter()
        try:
            counter.increment(-1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
        # Count should remain unchanged
        assert counter.count == 0


class TestMessageCounterThreadSafety:
    """Thread-safety tests."""

    def test_concurrent_increments_no_lost_counts(self) -> None:
        """Multiple threads incrementing should not lose any counts."""
        counter = MessageCounter()
        num_threads = 10
        increments_per_thread = 100
        barrier = threading.Barrier(num_threads)

        def worker() -> None:
            barrier.wait()  # Synchronize start
            for _ in range(increments_per_thread):
                counter.increment()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * increments_per_thread * 2  # default increment is 2
        assert counter.count == expected

    def test_concurrent_mixed_increments(self) -> None:
        """Threads with different increment values should sum correctly."""
        counter = MessageCounter()
        num_threads = 8
        barrier = threading.Barrier(num_threads)

        def worker(n: int) -> None:
            barrier.wait()
            for _ in range(50):
                counter.increment(n)

        threads = [threading.Thread(target=worker, args=(i + 1,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread i increments by (i+1) 50 times
        expected = sum((i + 1) * 50 for i in range(num_threads))
        assert counter.count == expected
