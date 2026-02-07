"""Tests for DQN replay buffer optimizations."""

from __future__ import annotations

import time

import numpy as np
import torch

from agents.dqn.replay_buffer import ReplayBuffer


class TestReplayBufferBasics:
    """Basic functionality tests."""

    def test_push_and_len(self, observation_shape: tuple[int, int, int]) -> None:
        """Buffer length should track pushed items."""
        buffer = ReplayBuffer(capacity=100, observation_shape=observation_shape)

        assert len(buffer) == 0

        state = np.zeros(observation_shape, dtype=np.float32)
        next_state = np.zeros(observation_shape, dtype=np.float32)

        for i in range(50):
            buffer.push(state, 0, 1.0, next_state, False)
            assert len(buffer) == i + 1

    def test_circular_overwrite(self, observation_shape: tuple[int, int, int]) -> None:
        """Buffer should overwrite oldest items when full."""
        capacity = 10
        buffer = ReplayBuffer(capacity=capacity, observation_shape=observation_shape)

        next_state = np.zeros(observation_shape, dtype=np.float32)

        # Fill buffer with distinct values
        for i in range(capacity):
            state = np.full(observation_shape, i, dtype=np.float32)
            buffer.push(state, i, float(i), next_state, False)

        assert len(buffer) == capacity

        # Overwrite first item
        new_state = np.full(observation_shape, 999, dtype=np.float32)
        buffer.push(new_state, 999, 999.0, next_state, False)

        assert len(buffer) == capacity
        # First position should have new value
        assert buffer.states[0].flat[0] == 999
        assert buffer.actions[0] == 999


class TestReplayBufferSampling:
    """Sampling behavior tests."""

    def test_sample_batch_shapes(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Sampled batch should have correct shapes."""
        buffer = ReplayBuffer(capacity=100, observation_shape=observation_shape)

        state = np.zeros(observation_shape, dtype=np.float32)
        next_state = np.zeros(observation_shape, dtype=np.float32)

        for _ in range(50):
            buffer.push(state, 0, 1.0, next_state, False)

        batch = buffer.sample(32, device)

        assert batch["states"].shape == (32, *observation_shape)
        assert batch["actions"].shape == (32,)
        assert batch["rewards"].shape == (32,)
        assert batch["next_states"].shape == (32, *observation_shape)
        assert batch["dones"].shape == (32,)

    def test_sample_values_correct(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Sampled values should match stored values."""
        buffer = ReplayBuffer(capacity=100, observation_shape=observation_shape)

        # Push one item with known values
        state = np.full(observation_shape, 1.5, dtype=np.float32)
        next_state = np.full(observation_shape, 2.5, dtype=np.float32)
        buffer.push(state, 2, 3.5, next_state, True)

        batch = buffer.sample(1, device)

        assert torch.allclose(batch["states"], torch.full((1, *observation_shape), 1.5))
        assert batch["actions"][0].item() == 2
        assert batch["rewards"][0].item() == 3.5
        assert torch.allclose(
            batch["next_states"], torch.full((1, *observation_shape), 2.5)
        )
        assert batch["dones"][0].item() == 1.0

    def test_sample_distribution(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """All items should have roughly equal sampling probability."""
        buffer = ReplayBuffer(capacity=100, observation_shape=observation_shape)

        next_state = np.zeros(observation_shape, dtype=np.float32)

        # Fill with distinct actions
        for i in range(100):
            state = np.zeros(observation_shape, dtype=np.float32)
            buffer.push(state, i % 4, 0.0, next_state, False)

        # Sample many times
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _ in range(1000):
            batch = buffer.sample(100, device)
            for action in batch["actions"].cpu().numpy():
                action_counts[int(action)] += 1

        # Each action should appear roughly 25% of the time
        total = sum(action_counts.values())
        for action, count in action_counts.items():
            ratio = count / total
            assert 0.20 < ratio < 0.30, f"Action {action} ratio {ratio} out of range"

    def test_sample_smaller_than_buffer(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Should handle batch_size > buffer size."""
        buffer = ReplayBuffer(capacity=100, observation_shape=observation_shape)

        state = np.zeros(observation_shape, dtype=np.float32)
        next_state = np.zeros(observation_shape, dtype=np.float32)

        for _ in range(10):
            buffer.push(state, 0, 1.0, next_state, False)

        # Request more than available
        batch = buffer.sample(100, device)

        # Should return what's available
        assert batch["states"].shape[0] == 10


class TestReplayBufferPerformance:
    """Performance benchmarks."""

    def test_sample_performance(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Sampling should be fast."""
        buffer = ReplayBuffer(capacity=100000, observation_shape=observation_shape)

        state = np.random.randn(*observation_shape).astype(np.float32)
        next_state = np.random.randn(*observation_shape).astype(np.float32)

        # Fill buffer
        for _ in range(10000):
            buffer.push(state, 0, 1.0, next_state, False)

        # Warm up
        for _ in range(10):
            buffer.sample(64, device)

        # Benchmark
        start = time.perf_counter()
        for _ in range(1000):
            buffer.sample(64, device)
        elapsed = time.perf_counter() - start

        # Should be well under 500ms for 1000 samples
        assert elapsed < 0.5, f"1000 samples took {elapsed*1000:.1f}ms"

    def test_push_performance(self, observation_shape: tuple[int, int, int]) -> None:
        """Pushing should be fast."""
        buffer = ReplayBuffer(capacity=100000, observation_shape=observation_shape)

        state = np.random.randn(*observation_shape).astype(np.float32)
        next_state = np.random.randn(*observation_shape).astype(np.float32)

        # Warm up
        for _ in range(100):
            buffer.push(state, 0, 1.0, next_state, False)

        # Benchmark
        start = time.perf_counter()
        for _ in range(10000):
            buffer.push(state, 0, 1.0, next_state, False)
        elapsed = time.perf_counter() - start

        # Should be well under 100ms for 10000 pushes
        assert elapsed < 0.1, f"10000 pushes took {elapsed*1000:.1f}ms"
