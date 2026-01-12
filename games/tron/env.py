"""Tron Light Cycles environment wrapping the core game logic."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from game.tron_state import Direction, TronState
from games.base import BaseGameEnv, GameMetadata
from games.registry import GameRegistry


class TronAgent(Protocol):
    """Protocol for Tron opponent agents."""

    def get_action(self, observation: np.ndarray) -> int:
        """Get action from observation."""
        ...


class RandomAgent:
    """Simple random agent for testing."""

    def get_action(self, observation: np.ndarray) -> int:
        """Return a random valid action."""
        return np.random.randint(0, 4)


class TronEnv(BaseGameEnv):
    """Gymnasium-compatible Tron Light Cycles environment.

    Single-agent perspective: controls player 1, opponent controls player 2.
    Supports self-play via set_opponent() method.
    """

    ACTION_MAP = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}
    ACTION_NAMES = ["up", "down", "left", "right"]

    def __init__(self, width: int = 20, height: int = 20, step_penalty: float = -0.01):
        """Initialize the Tron environment.

        Args:
            width: Grid width
            height: Grid height
            step_penalty: Small negative reward each step to encourage faster wins
        """
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self._state: TronState = TronState.create(width=width, height=height)
        self._opponent: TronAgent = RandomAgent()
        self._max_steps = width * height  # Max possible moves before filled

        # Observation: 4 channels (p1_trail, p2_trail, p1_head, p2_head)
        self.metadata = GameMetadata(
            name="tron",
            action_space_size=4,
            action_names=self.ACTION_NAMES,
            observation_shape=(4, height, width),
            max_episode_steps=self._max_steps,
            supports_mcts=True,
        )

    def set_opponent(self, agent: TronAgent) -> None:
        """Set the opponent agent for player 2.

        Args:
            agent: An agent implementing get_action(observation) -> int
        """
        self._opponent = agent

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the game to initial state.

        Args:
            seed: Optional random seed for reproducibility

        Returns:
            observation: Initial observation as numpy array
            info: Additional information dict
        """
        if seed is not None:
            np.random.seed(seed)

        self._state = TronState.create(width=self.width, height=self.height)

        obs = self._get_observation()
        info = {"turn": 0, "winner": None}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        Player 1 takes the given action, opponent takes its own action.

        Args:
            action: Integer action index for player 1

        Returns:
            observation: New observation
            reward: Reward for this step (player 1 perspective)
            terminated: Episode ended naturally (win/lose/draw)
            truncated: Episode ended due to step limit
            info: Additional info (turn, winner, etc.)
        """
        if self._state.is_terminal():
            # Game already over
            return (
                self._get_observation(),
                0.0,
                True,
                False,
                {"turn": self._state.turn, "winner": self._state.winner},
            )

        # Get opponent's observation (from player 2's perspective) and action
        opponent_obs = self._get_observation(player_perspective=2)
        opponent_action = self._opponent.get_action(opponent_obs)

        # Execute both actions simultaneously
        self._state = self._state.step(action, opponent_action)

        # Calculate reward from player 1's perspective
        reward = self.step_penalty  # Small step penalty
        terminated = self._state.is_terminal()
        truncated = False

        if terminated:
            winner = self._state.get_winner()
            if winner == 1:
                reward = 1.0  # Player 1 wins
            elif winner == 2:
                reward = -1.0  # Player 1 loses
            else:
                reward = 0.0  # Draw

        # Check for truncation (shouldn't happen with proper collision detection)
        if self._state.turn >= self._max_steps and not terminated:
            truncated = True

        obs = self._get_observation()
        info = {"turn": self._state.turn, "winner": self._state.winner}

        return obs, reward, terminated, truncated, info

    def get_valid_actions(self) -> list[int]:
        """Return list of valid action indices.

        In Tron, all 4 directions are technically valid moves,
        though some may lead to immediate death.
        """
        return [0, 1, 2, 3]

    def clone(self) -> TronEnv:
        """Create a deep copy of the environment for MCTS simulation."""
        new_env = TronEnv(
            width=self.width,
            height=self.height,
            step_penalty=self.step_penalty,
        )
        new_env._state = self._state.copy()
        new_env._opponent = self._opponent
        return new_env

    def render_state(self) -> dict[str, Any]:
        """Return state dict for WebSocket rendering.

        Returns dict with both players' trails and heads.
        """
        return self._state.to_dict()

    def get_observation(self) -> np.ndarray:
        """Get current observation without stepping."""
        return self._get_observation()

    def _get_observation(self, player_perspective: int = 1) -> np.ndarray:
        """Convert game state to neural network input.

        Returns 4-channel grid:
        - Channel 0: Player 1 trail (1 where trail exists)
        - Channel 1: Player 2 trail (1 where trail exists)
        - Channel 2: Player 1 head (1 at head position)
        - Channel 3: Player 2 head (1 at head position)

        When player_perspective=2, the channels are swapped so the
        opponent sees itself as "player 1" in the observation.

        Args:
            player_perspective: 1 for player 1's view, 2 for player 2's view
        """
        obs = np.zeros((4, self.height, self.width), dtype=np.float32)

        p1 = self._state.player1
        p2 = self._state.player2

        if player_perspective == 1:
            own_player, opp_player = p1, p2
        else:
            own_player, opp_player = p2, p1

        # Own trail (channel 0)
        for x, y in own_player.trail:
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[0, y, x] = 1.0

        # Opponent trail (channel 1)
        for x, y in opp_player.trail:
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[1, y, x] = 1.0

        # Own head (channel 2)
        if own_player.alive:
            hx, hy = own_player.x, own_player.y
            if 0 <= hx < self.width and 0 <= hy < self.height:
                obs[2, hy, hx] = 1.0

        # Opponent head (channel 3)
        if opp_player.alive:
            hx, hy = opp_player.x, opp_player.y
            if 0 <= hx < self.width and 0 <= hy < self.height:
                obs[3, hy, hx] = 1.0

        return obs

    def get_state(self) -> TronState:
        """Get the underlying game state (for testing/debugging)."""
        return self._state


# Register the game
GameRegistry.register("tron", TronEnv)
