from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GameState:
    """Represents the current state of a Snake game."""

    snake: List[Tuple[int, int]]  # List of (x, y) tuples, head first
    food: Tuple[int, int]  # (x, y) position of food
    direction: str  # Current direction: "up", "down", "left", "right"
    score: int  # Current score
    game_over: bool  # Whether the game has ended
    width: int  # Board width
    height: int  # Board height

    def to_dict(self) -> dict:
        """Convert GameState to a dictionary for JSON serialization."""
        return {
            "snake": [{"x": x, "y": y} for x, y in self.snake],
            "food": {"x": self.food[0], "y": self.food[1]},
            "direction": self.direction,
            "score": self.score,
            "game_over": self.game_over,
            "width": self.width,
            "height": self.height,
        }
