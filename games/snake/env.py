"""Snake environment wrapping existing game engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from game.engine import SnakeGame
from games.base import BaseGameEnv, GameMetadata
from games.registry import GameRegistry


class SnakeEnv(BaseGameEnv):
    """Gymnasium-compatible Snake environment.

    Wraps the existing SnakeGame engine with a standardized interface.
    """

    ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}
    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self._game = SnakeGame(width=width, height=height)
        self._step_count = 0
        self._max_steps = width * height * 10  # Reasonable limit

        # Pre-allocate observation buffer to avoid repeated allocation
        self._obs_buffer = np.zeros((3, height, width), dtype=np.float32)

        # Observation: 3 channels (snake body, snake head, food)
        self.metadata = GameMetadata(
            name="snake",
            action_space_size=4,
            action_names=self.ACTION_NAMES,
            observation_shape=(3, height, width),
            max_episode_steps=self._max_steps,
            supports_mcts=True,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the game."""
        if seed is not None:
            np.random.seed(seed)

        self._game.reset()
        self._step_count = 0

        obs = self._get_observation()
        info = {"score": 0}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step."""
        action_str = self.ACTION_MAP.get(action, "right")
        state, reward, done = self._game.step(action_str)

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        info = {"score": state.score, "length": len(state.snake)}

        return obs, reward, done, truncated, info

    def get_valid_actions(self) -> list[int]:
        """All 4 directions are always 'valid' (game handles invalid moves)."""
        return [0, 1, 2, 3]

    def clone(self) -> SnakeEnv:
        """Create copy for MCTS simulation.

        Uses optimized SnakeGame.copy() instead of copy.deepcopy()
        for ~10x faster cloning during MCTS tree search.
        """
        new_env = object.__new__(SnakeEnv)
        new_env.width = self.width
        new_env.height = self.height
        new_env._game = self._game.copy()
        new_env._step_count = self._step_count
        new_env._max_steps = self._max_steps
        # Pre-allocate new buffer for the cloned env
        new_env._obs_buffer = np.zeros((3, self.height, self.width), dtype=np.float32)
        new_env.metadata = self.metadata  # Shared reference is fine (immutable)
        return new_env

    def render_state(self) -> dict[str, Any]:
        """Return state for WebSocket rendering."""
        return self._game.get_state().to_dict()

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Convert game state to neural network input.

        Returns 3-channel grid:
        - Channel 0: Snake body (1 where body exists)
        - Channel 1: Snake head (1 at head position)
        - Channel 2: Food (1 at food position)

        Uses pre-allocated buffer and direct game attribute access
        to avoid allocation overhead and GameState creation.
        """
        # Clear buffer instead of allocating new array
        self._obs_buffer.fill(0)

        # Access game attributes directly (skip get_state() allocation)
        snake = self._game.snake
        food = self._game.food

        # Snake body (excluding head)
        for x, y in snake[1:]:
            if 0 <= x < self.width and 0 <= y < self.height:
                self._obs_buffer[0, y, x] = 1.0

        # Snake head
        if snake:
            hx, hy = snake[0]
            if 0 <= hx < self.width and 0 <= hy < self.height:
                self._obs_buffer[1, hy, hx] = 1.0

        # Food
        fx, fy = food
        if 0 <= fx < self.width and 0 <= fy < self.height:
            self._obs_buffer[2, fy, fx] = 1.0

        # Return copy to maintain caller independence
        return self._obs_buffer.copy()


# Register the game
GameRegistry.register("snake", SnakeEnv)
