"""Core game logic for Tron Light Cycles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Direction(IntEnum):
    """Movement directions."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# Direction vectors: (dx, dy)
DIRECTION_VECTORS: dict[Direction, tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}


@dataclass
class Player:
    """Represents a Tron player."""

    x: int
    y: int
    direction: Direction
    trail: set[tuple[int, int]] = field(default_factory=set)
    alive: bool = True

    def copy(self) -> Player:
        """Create a deep copy of this player."""
        return Player(
            x=self.x,
            y=self.y,
            direction=self.direction,
            trail=set(self.trail),
            alive=self.alive,
        )


@dataclass
class TronState:
    """Represents the current state of a Tron Light Cycles game.

    Two players move on a grid, leaving trails behind them.
    A player loses when they collide with a wall, their own trail,
    or the opponent's trail.
    """

    player1: Player
    player2: Player
    width: int
    height: int
    game_over: bool = False
    winner: int | None = None  # 1, 2, or None (draw)
    turn: int = 0

    @classmethod
    def create(cls, width: int = 20, height: int = 20) -> TronState:
        """Create a new Tron game state with default starting positions."""
        # Player 1 starts on the left side, facing right
        p1_x, p1_y = width // 4, height // 2
        player1 = Player(
            x=p1_x,
            y=p1_y,
            direction=Direction.RIGHT,
            trail={(p1_x, p1_y)},
        )

        # Player 2 starts on the right side, facing left
        p2_x, p2_y = (3 * width) // 4, height // 2
        player2 = Player(
            x=p2_x,
            y=p2_y,
            direction=Direction.LEFT,
            trail={(p2_x, p2_y)},
        )

        return cls(
            player1=player1,
            player2=player2,
            width=width,
            height=height,
        )

    def reset(self) -> TronState:
        """Reset the game to initial state."""
        return TronState.create(self.width, self.height)

    def copy(self) -> TronState:
        """Create a deep copy of this state."""
        return TronState(
            player1=self.player1.copy(),
            player2=self.player2.copy(),
            width=self.width,
            height=self.height,
            game_over=self.game_over,
            winner=self.winner,
            turn=self.turn,
        )

    def _is_collision(self, x: int, y: int, player_idx: int) -> bool:
        """Check if position causes a collision.

        Args:
            x: X coordinate to check
            y: Y coordinate to check
            player_idx: 1 or 2, the player trying to move here

        Returns:
            True if this position would cause a collision
        """
        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Collision with either trail (including own trail)
        if (x, y) in self.player1.trail or (x, y) in self.player2.trail:
            return True

        return False

    def step(self, action1: int, action2: int) -> TronState:
        """Execute one step for both players simultaneously.

        Args:
            action1: Direction for player 1 (0=up, 1=down, 2=left, 3=right)
            action2: Direction for player 2 (0=up, 1=down, 2=left, 3=right)

        Returns:
            New game state after the move
        """
        if self.game_over:
            return self

        new_state = self.copy()
        new_state.turn += 1

        # Update directions
        new_state.player1.direction = Direction(action1)
        new_state.player2.direction = Direction(action2)

        # Calculate new positions
        dx1, dy1 = DIRECTION_VECTORS[new_state.player1.direction]
        new_x1 = new_state.player1.x + dx1
        new_y1 = new_state.player1.y + dy1

        dx2, dy2 = DIRECTION_VECTORS[new_state.player2.direction]
        new_x2 = new_state.player2.x + dx2
        new_y2 = new_state.player2.y + dy2

        # Check for collisions
        p1_collision = new_state._is_collision(new_x1, new_y1, 1)
        p2_collision = new_state._is_collision(new_x2, new_y2, 2)

        # Check for head-to-head collision
        head_collision = new_x1 == new_x2 and new_y1 == new_y2

        # Determine outcome
        if head_collision or (p1_collision and p2_collision):
            # Both collide - draw
            new_state.player1.alive = False
            new_state.player2.alive = False
            new_state.game_over = True
            new_state.winner = None
        elif p1_collision:
            # Player 1 loses
            new_state.player1.alive = False
            new_state.game_over = True
            new_state.winner = 2
        elif p2_collision:
            # Player 2 loses
            new_state.player2.alive = False
            new_state.game_over = True
            new_state.winner = 1
        else:
            # Both survive - update positions and trails
            new_state.player1.x = new_x1
            new_state.player1.y = new_y1
            new_state.player1.trail.add((new_x1, new_y1))

            new_state.player2.x = new_x2
            new_state.player2.y = new_y2
            new_state.player2.trail.add((new_x2, new_y2))

        return new_state

    def is_terminal(self) -> bool:
        """Check if the game has ended."""
        return self.game_over

    def get_winner(self) -> int | None:
        """Get the winner (1, 2, or None for draw/ongoing)."""
        return self.winner

    def to_dict(self) -> dict[str, Any]:
        """Convert TronState to a dictionary for JSON serialization."""
        return {
            "player1": {
                "x": self.player1.x,
                "y": self.player1.y,
                "direction": self.player1.direction.name.lower(),
                "trail": [{"x": x, "y": y} for x, y in self.player1.trail],
                "alive": self.player1.alive,
            },
            "player2": {
                "x": self.player2.x,
                "y": self.player2.y,
                "direction": self.player2.direction.name.lower(),
                "trail": [{"x": x, "y": y} for x, y in self.player2.trail],
                "alive": self.player2.alive,
            },
            "width": self.width,
            "height": self.height,
            "game_over": self.game_over,
            "winner": self.winner,
            "turn": self.turn,
        }
