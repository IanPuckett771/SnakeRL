"""Tests for encode_state: Rust vs Python equivalence, correctness, and performance."""

import time

import numpy as np
import pytest

from algorithms.base import _encode_state_py, encode_state
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


class TestEncodeStateCorrectness:
    """Basic correctness tests for the 44-feature flat encoder."""

    def test_output_shape(self):
        state = _make_state()
        features = encode_state(state)
        assert features.shape == (44,)
        assert features.dtype == np.float32

    def test_head_position(self):
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)])
        f = encode_state(state)
        assert f[0] == pytest.approx(5 / 20)
        assert f[1] == pytest.approx(5 / 20)

    def test_food_direction(self):
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)], food=(10, 15))
        f = encode_state(state)
        assert f[2] == pytest.approx((10 - 5) / 20)
        assert f[3] == pytest.approx((15 - 5) / 20)

    def test_direction_onehot(self):
        for direction, idx in [("up", 0), ("down", 1), ("left", 2), ("right", 3)]:
            state = _make_state(direction=direction)
            f = encode_state(state)
            for j in range(4):
                expected = 1.0 if j == idx else 0.0
                assert f[4 + j] == pytest.approx(expected), \
                    f"direction={direction}, index={4+j}"

    def test_danger_blocked(self):
        """Snake at (0, 0) facing right — up and left should be danger."""
        state = _make_state(snake=[(0, 0), (1, 0)], walls=[], direction="right")
        f = encode_state(state)
        # Directions: up(0,-1), down(0,1), left(-1,0), right(1,0)
        assert f[8] == 1.0   # up: out of bounds
        assert f[10] == 1.0  # left: out of bounds

    def test_danger_safe(self):
        """Snake at (5, 5) — all cardinal directions should be safe on empty board."""
        state = _make_state(snake=[(5, 5), (4, 5)], walls=[])
        f = encode_state(state)
        # up, down safe (no body there)
        assert f[8] == 0.0   # up
        assert f[9] == 0.0   # down
        # right safe
        assert f[11] == 0.0  # right

    def test_ray_wall_distance_north(self):
        """Ray N from (5, 5) on 20x20 board with no walls — should hit y=0 boundary."""
        state = _make_state(snake=[(5, 5), (4, 5)], walls=[])
        f = encode_state(state)
        # Ray 0 (N): dist_wall = 5 steps to boundary (y goes 4,3,2,1,0,-1 → 5+1=6? No: y=4,3,2,1,0 = 5 steps, then -1 OOB at step 6)
        # Actually: from (5,5), N ray: step1→(5,4), step2→(5,3), step3→(5,2), step4→(5,1), step5→(5,0), step6→(5,-1) OOB
        # dist_wall = 6 / 20 = 0.3
        expected_dist = 6 / 20  # 6 steps to go out of bounds
        assert f[16] == pytest.approx(expected_dist, abs=1e-5)

    def test_ray_wall_distance_east(self):
        """Ray E from (5, 5) on 20x20 board with no walls."""
        state = _make_state(snake=[(5, 5), (4, 5)], walls=[])
        f = encode_state(state)
        # Ray 2 (E): steps from (5,5) going right: (6,5)...(19,5),(20,5) OOB at step 15
        expected_dist = 15 / 20
        assert f[16 + 2 * 3] == pytest.approx(expected_dist, abs=1e-5)

    def test_ray_body_detection(self):
        """Ray should detect body segments."""
        # Snake goes right: head at (10,5), body at (9,5), (8,5), (7,5)
        # Ray W from (10,5) should hit body at step 1
        state = _make_state(
            snake=[(10, 5), (9, 5), (8, 5), (7, 5)],
            walls=[],
        )
        f = encode_state(state)
        # Ray 6 (W): step1→(9,5) which is body
        ray_w_body = f[16 + 6 * 3 + 1]  # dist_body for W ray
        assert ray_w_body == pytest.approx(1 / 20, abs=1e-5)

    def test_ray_food_detection(self):
        """Ray should detect food along its path."""
        # Food at (10, 5), head at (5, 5) — ray E should detect food at step 5
        state = _make_state(
            snake=[(5, 5), (4, 5)],
            food=(10, 5),
            walls=[],
        )
        f = encode_state(state)
        # Ray 2 (E): step5→(10,5) which is food
        ray_e_food = f[16 + 2 * 3 + 2]  # dist_food for E ray
        assert ray_e_food == pytest.approx(5 / 20, abs=1e-5)

    def test_ray_no_body_is_1(self):
        """If no body along a ray, dist_body should be 1.0."""
        state = _make_state(snake=[(5, 5), (4, 5)], walls=[])
        f = encode_state(state)
        # Ray 0 (N): no body in that direction
        ray_n_body = f[16 + 0 * 3 + 1]
        assert ray_n_body == pytest.approx(1.0)

    def test_ray_no_food_is_1(self):
        """If food is not along a ray, dist_food should be 1.0."""
        state = _make_state(snake=[(5, 5), (4, 5)], food=(15, 15), walls=[])
        f = encode_state(state)
        # Ray 0 (N): food is at (15,15), not on N ray
        ray_n_food = f[16 + 0 * 3 + 2]
        assert ray_n_food == pytest.approx(1.0)

    def test_snake_length(self):
        snake = [(5, 5), (4, 5), (3, 5), (2, 5)]
        state = _make_state(snake=snake, walls=[])
        f = encode_state(state)
        assert f[40] == pytest.approx(4 / 400)

    def test_tail_direction(self):
        state = _make_state(snake=[(5, 5), (4, 5), (3, 5)], walls=[])
        f = encode_state(state)
        # tail at (3, 5), head at (5, 5)
        assert f[41] == pytest.approx((3 - 5) / 20)
        assert f[42] == pytest.approx(0.0)

    def test_body_density_ahead(self):
        # Head at (5,5), direction=right, body at (6,5), (7,5) — both ahead
        state = _make_state(
            snake=[(5, 5), (6, 5), (7, 5)],
            direction="right",
            walls=[],
        )
        f = encode_state(state)
        # body_ahead: (6,5) has dot=(1)*1+(0)*0=1>0, (7,5) has dot=2>0. Both ahead.
        assert f[43] == pytest.approx(2 / 2)

    def test_small_board(self):
        state = _make_state(
            snake=[(1, 1), (0, 1)],
            width=3, height=3,
            walls=[],
            food=(2, 2),
        )
        f = encode_state(state)
        assert f.shape == (44,)

    def test_all_directions(self):
        for direction in ["up", "down", "left", "right"]:
            state = _make_state(direction=direction)
            f = encode_state(state)
            assert f.shape == (44,)


class TestEncodeStateEquivalence:
    """Ensure Rust and Python implementations produce identical output."""

    def _assert_equivalent(self, state):
        py_features = _encode_state_py(state)
        rs_features = encode_state(state)
        np.testing.assert_allclose(
            rs_features, py_features, atol=1e-6,
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

    def test_food_on_ray(self):
        """Food directly along a ray path."""
        self._assert_equivalent(
            _make_state(snake=[(5, 5), (4, 5)], food=(5, 0), walls=[])
        )

    def test_body_on_ray(self):
        """Body segments along multiple ray paths."""
        snake = [(5, 5), (5, 6), (5, 7), (6, 7), (7, 7)]
        self._assert_equivalent(_make_state(snake=snake, walls=[]))

    def test_small_board(self):
        self._assert_equivalent(
            _make_state(snake=[(1, 1), (0, 1)], width=3, height=3, walls=[], food=(2, 2))
        )

    def test_wall_on_ray(self):
        """Wall directly in the path of a ray."""
        self._assert_equivalent(
            _make_state(snake=[(5, 5), (4, 5)], walls=[(5, 3)])
        )


@pytest.mark.performance
class TestEncodeStatePerformance:
    """Benchmark Rust vs Python encode_state."""

    def test_benchmark(self):
        state = _make_state(
            snake=[(i, 5) for i in range(10, -1, -1)],
            walls=[(x, 0) for x in range(20)] + [(x, 19) for x in range(20)],
        )
        n = 10_000

        # Warm up
        for _ in range(100):
            encode_state(state)
            _encode_state_py(state)

        # Benchmark Python
        start = time.perf_counter()
        for _ in range(n):
            _encode_state_py(state)
        py_time = time.perf_counter() - start

        # Benchmark current (Rust if available, else Python)
        start = time.perf_counter()
        for _ in range(n):
            encode_state(state)
        rs_time = time.perf_counter() - start

        speedup = py_time / rs_time
        print(f"\n  Python: {py_time:.3f}s | Active: {rs_time:.3f}s | "
              f"Speedup: {speedup:.1f}x ({n} iterations)")
