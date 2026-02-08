"""Tests for Snake environment optimizations."""

from __future__ import annotations

import time

import numpy as np

from games.snake.env import SnakeEnv


class TestSnakeEnvClone:
    """Tests for SnakeEnv.clone() correctness and performance."""

    def test_clone_creates_independent_copy(self) -> None:
        """Cloned env should be independent of original."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        # Clone and modify original
        cloned = env.clone()
        original_snake = list(env._game.snake)
        original_food = env._game.food

        # Step original
        env.step(0)

        # Cloned should be unchanged
        assert cloned._game.snake == original_snake
        assert cloned._game.food == original_food

    def test_clone_copies_all_state(self) -> None:
        """Clone should copy all game state."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        # Advance game state
        for _ in range(10):
            env.step(3)  # Move right

        cloned = env.clone()

        # Verify all state is copied
        assert cloned._game.snake == env._game.snake
        assert cloned._game.food == env._game.food
        assert cloned._game.direction == env._game.direction
        assert cloned._game.score == env._game.score
        assert cloned._game.game_over == env._game.game_over
        assert cloned._game.walls == env._game.walls
        assert cloned._step_count == env._step_count
        assert cloned.width == env.width
        assert cloned.height == env.height

    def test_clone_snake_list_independence(self) -> None:
        """Modifying cloned snake list should not affect original."""
        env = SnakeEnv(width=20, height=20)
        env.reset()
        cloned = env.clone()

        # Modify cloned snake list
        original_len = len(env._game.snake)
        cloned._game.snake.append((0, 0))

        # Original should be unchanged
        assert len(env._game.snake) == original_len

    def test_clone_walls_list_independence(self) -> None:
        """Modifying cloned walls list should not affect original."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        # Get snake to length 5+ to spawn walls
        env._game.snake = [(10, 10), (9, 10), (8, 10), (7, 10), (6, 10)]
        env._game._spawn_walls()

        cloned = env.clone()
        original_walls = list(env._game.walls)

        # Modify cloned walls
        cloned._game.walls.append((0, 0))

        # Original should be unchanged
        assert env._game.walls == original_walls

    def test_clone_performance(self) -> None:
        """Clone should be fast enough for MCTS (1000 clones < 50ms)."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        # Warm up
        for _ in range(10):
            env.clone()

        # Benchmark
        start = time.perf_counter()
        for _ in range(1000):
            env.clone()
        elapsed = time.perf_counter() - start

        # Should be well under 50ms for 1000 clones
        assert elapsed < 0.05, f"1000 clones took {elapsed*1000:.1f}ms, expected < 50ms"


class TestSnakeEnvObservation:
    """Tests for observation generation."""

    def test_observation_shape(self) -> None:
        """Observation should have correct shape."""
        env = SnakeEnv(width=20, height=20)
        obs, _ = env.reset()
        assert obs.shape == (3, 20, 20)
        assert obs.dtype == np.float32

    def test_observation_values(self) -> None:
        """Observation channels should have correct values."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        # Get observation and game state
        obs = env._get_observation()
        snake = env._game.snake
        food = env._game.food

        # Check snake body channel
        for x, y in snake[1:]:
            assert obs[0, y, x] == 1.0, f"Snake body at ({x}, {y}) not marked"

        # Check snake head channel
        hx, hy = snake[0]
        assert obs[1, hy, hx] == 1.0, f"Snake head at ({hx}, {hy}) not marked"

        # Check food channel
        fx, fy = food
        assert obs[2, fy, fx] == 1.0, f"Food at ({fx}, {fy}) not marked"

    def test_observation_returns_copy(self) -> None:
        """Observation should return a copy, not the buffer."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        obs1 = env._get_observation()
        obs2 = env._get_observation()

        # Modify obs1
        obs1[0, 0, 0] = 999.0

        # obs2 should be unaffected
        assert obs2[0, 0, 0] != 999.0

    def test_observation_clears_between_calls(self) -> None:
        """Observation buffer should be cleared between calls."""
        env = SnakeEnv(width=20, height=20)
        env.reset()

        env._get_observation()
        snake_positions_1 = set()
        for x, y in env._game.snake:
            snake_positions_1.add((x, y))

        # Move snake
        env.step(3)  # Move right

        obs2 = env._get_observation()
        snake_positions_2 = set()
        for x, y in env._game.snake:
            snake_positions_2.add((x, y))

        # Old positions should be cleared (unless still occupied)
        for x, y in snake_positions_1 - snake_positions_2:
            if 0 <= x < 20 and 0 <= y < 20:
                assert obs2[0, y, x] == 0.0 or obs2[1, y, x] == 0.0
