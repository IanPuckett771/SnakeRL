"""Snake environment wrapping existing game engine."""

from __future__ import annotations

import copy
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
        """Create deep copy for MCTS simulation."""
        new_env = SnakeEnv(width=self.width, height=self.height)
        new_env._game = copy.deepcopy(self._game)
        new_env._step_count = self._step_count
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
        """
        obs = np.zeros((3, self.height, self.width), dtype=np.float32)
        state = self._game.get_state()

        # Snake body (excluding head)
        for x, y in state.snake[1:]:
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[0, y, x] = 1.0

        # Snake head
        if state.snake:
            hx, hy = state.snake[0]
            if 0 <= hx < self.width and 0 <= hy < self.height:
                obs[1, hy, hx] = 1.0

        # Food
        fx, fy = state.food
        if 0 <= fx < self.width and 0 <= fy < self.height:
            obs[2, fy, fx] = 1.0

        return obs


# Register the game
GameRegistry.register("snake", SnakeEnv)
