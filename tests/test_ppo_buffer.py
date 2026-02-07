"""Tests for PPO rollout buffer optimizations."""

from __future__ import annotations

import time

import numpy as np
import torch

from agents.ppo.buffer import HAS_NUMBA, RolloutBuffer


class TestPPOBufferGAE:
    """Tests for GAE computation correctness."""

    def test_gae_basic_computation(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """GAE should compute correct values for simple case."""
        buffer = RolloutBuffer(
            buffer_size=5,
            observation_shape=observation_shape,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        obs = np.zeros(observation_shape, dtype=np.float32)

        # Add some steps with known values
        buffer.add(obs, action=0, reward=1.0, done=False, value=0.5, log_prob=-1.0)
        buffer.add(obs, action=1, reward=1.0, done=False, value=0.5, log_prob=-1.0)
        buffer.add(obs, action=0, reward=1.0, done=False, value=0.5, log_prob=-1.0)

        buffer.compute_returns_and_advantages(last_value=0.5)

        # Verify advantages are computed
        assert not np.allclose(buffer.advantages[:3], 0.0)
        # Returns should equal advantages + values
        np.testing.assert_allclose(
            buffer.returns[:3], buffer.advantages[:3] + buffer.values[:3], rtol=1e-5
        )

    def test_gae_done_resets_advantage(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """GAE should reset when episode ends."""
        buffer = RolloutBuffer(
            buffer_size=10,
            observation_shape=observation_shape,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        obs = np.zeros(observation_shape, dtype=np.float32)

        # Episode 1: ends with done=True
        buffer.add(obs, 0, reward=1.0, done=False, value=1.0, log_prob=-1.0)
        buffer.add(obs, 0, reward=1.0, done=True, value=1.0, log_prob=-1.0)

        # Episode 2: continues
        buffer.add(obs, 0, reward=1.0, done=False, value=1.0, log_prob=-1.0)
        buffer.add(obs, 0, reward=1.0, done=False, value=1.0, log_prob=-1.0)

        buffer.compute_returns_and_advantages(last_value=1.0)

        # After done, next step's advantage should not depend on previous episode
        # The advantage at step 2 should be based only on steps 2-3
        # (This is a property check, not exact value check)
        assert buffer.advantages[2] != 0.0

    def test_gae_python_vs_numba_equivalence(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Python and numba implementations should produce identical results."""
        if not HAS_NUMBA:
            return  # Skip if numba not available

        # Create two identical buffers
        buffer_numba = RolloutBuffer(
            buffer_size=100,
            observation_shape=observation_shape,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )
        buffer_python = RolloutBuffer(
            buffer_size=100,
            observation_shape=observation_shape,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        np.random.seed(42)
        obs = np.zeros(observation_shape, dtype=np.float32)

        # Fill with random data
        for _ in range(50):
            reward = np.random.randn()
            value = np.random.randn()
            done = np.random.random() < 0.1
            log_prob = np.random.randn()

            buffer_numba.add(obs, 0, reward, done, value, log_prob)
            buffer_python.add(obs, 0, reward, done, value, log_prob)

        last_value = np.random.randn()

        # Compute with numba
        buffer_numba.compute_returns_and_advantages(last_value)

        # Force Python fallback
        buffer_python._compute_gae_python(last_value)

        # Should be numerically equivalent
        np.testing.assert_allclose(
            buffer_numba.advantages[:50],
            buffer_python.advantages[:50],
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            buffer_numba.returns[:50], buffer_python.returns[:50], rtol=1e-5, atol=1e-7
        )


class TestPPOBufferPerformance:
    """Performance benchmarks."""

    def test_gae_performance(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """GAE computation should be fast."""
        buffer_size = 2048
        buffer = RolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            device=device,
            gamma=0.99,
            gae_lambda=0.95,
        )

        obs = np.zeros(observation_shape, dtype=np.float32)

        # Fill buffer
        for i in range(buffer_size):
            done = i > 0 and i % 200 == 0
            buffer.add(obs, 0, reward=1.0, done=done, value=0.5, log_prob=-1.0)

        # Warm up (especially important for numba JIT)
        buffer.compute_returns_and_advantages(0.5)
        buffer.reset()
        for i in range(buffer_size):
            done = i > 0 and i % 200 == 0
            buffer.add(obs, 0, reward=1.0, done=done, value=0.5, log_prob=-1.0)

        # Benchmark
        start = time.perf_counter()
        for _ in range(1000):
            buffer.compute_returns_and_advantages(0.5)
        elapsed = time.perf_counter() - start

        # With numba: should be < 100ms for 1000 iterations
        # With Python: should be < 2000ms (still reasonable)
        if HAS_NUMBA:
            msg = f"1000 GAE computations took {elapsed*1000:.1f}ms with numba"
            assert elapsed < 0.1, msg
        else:
            msg = f"1000 GAE computations took {elapsed*1000:.1f}ms (Python fallback)"
            assert elapsed < 2.0, msg


class TestPPOBufferBatching:
    """Tests for batch generation."""

    def test_batch_shapes(
        self, observation_shape: tuple[int, int, int], device: torch.device
    ) -> None:
        """Generated batches should have correct shapes."""
        buffer = RolloutBuffer(
            buffer_size=100,
            observation_shape=observation_shape,
            device=device,
        )

        obs = np.zeros(observation_shape, dtype=np.float32)

        for _ in range(100):
            buffer.add(obs, 0, reward=1.0, done=False, value=0.5, log_prob=-1.0)

        buffer.compute_returns_and_advantages(0.5)

        for batch in buffer.get(batch_size=32):
            assert batch["observations"].shape == (32, *observation_shape)
            assert batch["actions"].shape == (32,)
            assert batch["old_log_probs"].shape == (32,)
            assert batch["advantages"].shape == (32,)
            assert batch["returns"].shape == (32,)
            break  # Just check first batch
