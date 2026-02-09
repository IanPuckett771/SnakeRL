"""Tests for encode_state_grid: Rust vs Python equivalence and correctness."""

import time

import numpy as np
import pytest

from algorithms.base import _encode_state_grid_py, encode_state_grid
from game.state import GameState


def _make_state(**kwargs):
    """Create a GameState with sensible defaults, overridable via kwargs."""
    defaults = dict(
        snake=[(5, 5), (4, 5), (3, 5), (2, 5)],
        food=(10, 10),
        food_points=5,
        walls=[(0, 0), (19, 19), (0, 19), (19, 0)],
        direction="right",
        width=20,
        height=20,
    )
    defaults.update(kwargs)
    return GameState(**defaults)


class TestEncodeStateGridCorrectness:
    """Basic correctness tests for the grid encoder."""

    def test_output_shape(self):
        state = _make_state()
        grid = encode_state_grid(state)
        assert grid.shape == (7, 20, 20)
        assert grid.dtype == np.float32

    def test_head_channel(self):
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)])
        grid = encode_state_grid(state)
        assert grid[0, 5, 5] == 1.0
        # No other cell should be 1.0 in head channel
        grid[0, 5, 5] = 0.0
        assert grid[0].sum() == 0.0

    def test_tail_channel(self):
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)])
        grid = encode_state_grid(state)
        assert grid[2, 5, 3] == 1.0

    def test_food_channel(self):
        state = _make_state(food=(10, 10), food_points=10)
        grid = encode_state_grid(state)
        assert grid[3, 10, 10] == pytest.approx(1.0)

    def test_wall_channel(self):
        state = _make_state(walls=[(1, 2), (3, 4)])
        grid = encode_state_grid(state)
        assert grid[4, 2, 1] == 1.0
        assert grid[4, 4, 3] == 1.0

    def test_no_walls(self):
        state = _make_state(walls=[])
        grid = encode_state_grid(state)
        assert grid[4].sum() == 0.0

    def test_reachability_head_always_reachable(self):
        state = _make_state()
        grid = encode_state_grid(state)
        hx, hy = state.snake[0]
        assert grid[6, hy, hx] == 1.0

    def test_reachability_body_blocked(self):
        """Body segments should be marked as blocked (0.0) in reachability."""
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)], walls=[])
        grid = encode_state_grid(state)
        # Body cells should not be reachable
        assert grid[6, 5, 4] == 0.0
        assert grid[6, 5, 3] == 0.0

    def test_small_board(self):
        state = _make_state(snake=[(1, 1), (0, 1)], width=3, height=3, walls=[], food=(2, 2))
        grid = encode_state_grid(state)
        assert grid.shape == (7, 3, 3)

    def test_all_directions(self):
        for direction in ["up", "down", "left", "right"]:
            state = _make_state(direction=direction)
            grid = encode_state_grid(state)
            assert grid.shape == (7, 20, 20)


class TestEncodeStateGridEquivalence:
    """Ensure Rust and Python implementations produce identical output."""

    def _assert_equivalent(self, state):
        py_grid = _encode_state_grid_py(state)
        rs_grid = encode_state_grid(state)
        np.testing.assert_allclose(
            rs_grid, py_grid, atol=1e-6,
            err_msg=f"Rust/Python mismatch for state: snake={state.snake}, "
                    f"direction={state.direction}",
        )

    def test_basic(self):
        self._assert_equivalent(_make_state())

    def test_no_walls(self):
        self._assert_equivalent(_make_state(walls=[]))

    def test_long_snake(self):
        snake = [(i, 5) for i in range(15, -1, -1)]
        self._assert_equivalent(_make_state(snake=snake))

    def test_length_2_snake(self):
        self._assert_equivalent(_make_state(snake=[(5, 5), (4, 5)]))

    def test_length_1_snake(self):
        self._assert_equivalent(_make_state(snake=[(5, 5)]))

    def test_all_directions(self):
        for d in ["up", "down", "left", "right"]:
            self._assert_equivalent(_make_state(direction=d))

    def test_corner_positions(self):
        for pos in [(0, 0), (19, 0), (0, 19), (19, 19)]:
            self._assert_equivalent(
                _make_state(snake=[pos, (pos[0], max(pos[1] - 1, 0))], walls=[])
            )

    def test_many_walls(self):
        walls = [(x, 0) for x in range(20)] + [(x, 19) for x in range(20)]
        self._assert_equivalent(_make_state(walls=walls))

    def test_high_food_points(self):
        self._assert_equivalent(_make_state(food_points=20))

    def test_small_board(self):
        self._assert_equivalent(
            _make_state(snake=[(1, 1), (0, 1)], width=3, height=3, walls=[], food=(2, 2))
        )


@pytest.mark.performance
class TestEncodeStateGridPerformance:
    """Benchmark Rust vs Python encode_state_grid."""

    def test_benchmark(self):
        state = _make_state(
            snake=[(i, 5) for i in range(10, -1, -1)],
            walls=[(x, 0) for x in range(20)] + [(x, 19) for x in range(20)],
        )
        n = 10_000

        # Warm up
        for _ in range(100):
            encode_state_grid(state)
            _encode_state_grid_py(state)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n):
            _encode_state_grid_py(state)
        py_time = time.perf_counter() - start

        # Benchmark current (Rust if available, else Python)
        start = time.perf_counter()
        for _ in range(n):
            encode_state_grid(state)
        rs_time = time.perf_counter() - start

        speedup = py_time / rs_time
        print(f"\n  Python: {py_time:.3f}s | Active: {rs_time:.3f}s | "
              f"Speedup: {speedup:.1f}x ({n} iterations)")
