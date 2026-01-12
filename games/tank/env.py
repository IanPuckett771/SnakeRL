"""Tank Battle environment wrapping TankState game engine."""

from __future__ import annotations

from typing import Any

import numpy as np

from game.tank_state import TankState
from games.base import BaseGameEnv, GameMetadata
from games.registry import GameRegistry


class TankEnv(BaseGameEnv):
    """Gymnasium-compatible Tank Battle environment.

    Wraps the TankState engine with a standardized interface.
    Now uses smooth physics with pixel coordinates internally,
    but provides grid-based observations for neural networks.

    Observation space: 5 channels (walls, player, enemies, bullets, collectibles)
    Action space: 5 actions (forward, backward, turn_left, turn_right, shoot)
    """

    ACTION_NAMES = ["forward", "backward", "turn_left", "turn_right", "shoot"]
    CELL_SIZE = 20  # Pixels per grid cell for observation conversion

    def __init__(self, width: int = 20, height: int = 20, num_enemies: int = 3):
        self.width = width
        self.height = height
        self.num_enemies = num_enemies
        self._state: TankState = TankState.create(
            width=width, height=height, num_enemies=num_enemies
        )
        self._step_count = 0
        self._max_steps = width * height * 10  # Reasonable limit

        # Observation: 5 channels (walls, player, enemies, bullets, collectibles)
        self.metadata = GameMetadata(
            name="tank",
            action_space_size=5,
            action_names=self.ACTION_NAMES,
            observation_shape=(5, height, width),
            max_episode_steps=self._max_steps,
            supports_mcts=True,
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the game."""
        if seed is not None:
            np.random.seed(seed)

        self._state = TankState.create(
            width=self.width, height=self.height, num_enemies=self.num_enemies
        )
        self._step_count = 0

        obs = self._get_observation()
        info = {"score": 0, "health": self._state.player.health}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step."""
        self._state, reward, done = self._state.step(action)

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        info = {
            "score": self._state.score,
            "health": self._state.player.health,
            "enemies_destroyed": self._state.enemies_destroyed,
            "turn": self._state.turn,
        }

        return obs, reward, done, truncated, info

    def get_valid_actions(self) -> list[int]:
        """All 5 actions are always 'valid' (game handles invalid moves)."""
        return [0, 1, 2, 3, 4]

    def clone(self) -> TankEnv:
        """Create deep copy for MCTS simulation."""
        new_env = TankEnv(width=self.width, height=self.height, num_enemies=self.num_enemies)
        new_env._state = self._state.copy()
        new_env._step_count = self._step_count
        return new_env

    def render_state(self) -> dict[str, Any]:
        """Return state for WebSocket rendering."""
        return self._state.to_dict()

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self._get_observation()

    def _pixel_to_grid(self, px: float, py: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid coordinates."""
        gx = int(px / self.CELL_SIZE)
        gy = int(py / self.CELL_SIZE)
        return (
            max(0, min(self.width - 1, gx)),
            max(0, min(self.height - 1, gy)),
        )

    def _get_observation(self) -> np.ndarray:
        """Convert game state to neural network input.

        Converts pixel coordinates to grid coordinates for observation.

        Returns 5-channel grid:
        - Channel 0: Walls (1 where wall exists)
        - Channel 1: Player tank (1 at player position)
        - Channel 2: Enemy tanks (1 at each enemy position)
        - Channel 3: Bullets (1 at each bullet position)
        - Channel 4: Collectibles (1 at each collectible position)
        """
        obs = np.zeros((5, self.height, self.width), dtype=np.float32)

        # Channel 0: Walls (now Wall objects with x, y, width, height)
        for wall in self._state.walls:
            # Fill in grid cells covered by wall
            x1 = int(wall.x / self.CELL_SIZE)
            y1 = int(wall.y / self.CELL_SIZE)
            x2 = int((wall.x + wall.width) / self.CELL_SIZE)
            y2 = int((wall.y + wall.height) / self.CELL_SIZE)
            for gx in range(max(0, x1), min(self.width, x2 + 1)):
                for gy in range(max(0, y1), min(self.height, y2 + 1)):
                    obs[0, gy, gx] = 1.0

        # Channel 1: Player tank
        if self._state.player.alive:
            gx, gy = self._pixel_to_grid(self._state.player.x, self._state.player.y)
            obs[1, gy, gx] = 1.0

        # Channel 2: Enemy tanks
        for enemy in self._state.enemies:
            if enemy.alive:
                gx, gy = self._pixel_to_grid(enemy.x, enemy.y)
                obs[2, gy, gx] = 1.0

        # Channel 3: Bullets
        for bullet in self._state.bullets:
            gx, gy = self._pixel_to_grid(bullet.x, bullet.y)
            obs[3, gy, gx] = 1.0

        # Channel 4: Collectibles
        for coll in self._state.collectibles:
            gx, gy = self._pixel_to_grid(coll.x, coll.y)
            obs[4, gy, gx] = 1.0

        return obs


# Register the game
GameRegistry.register("tank", TankEnv)
