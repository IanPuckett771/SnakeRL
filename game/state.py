from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GameState:
    """Represents the current state of a Snake game."""

    snake: List[Tuple[int, int]]  # List of (x, y) tuples, head first
    food: Tuple[int, int]  # (x, y) position of food
    food_type: str = "red"  # Color/type of food: "red", "orange", "yellow", "green", "blue"
    food_points: int = 1  # Points this food is worth
    walls: List[Tuple[int, int]] = field(default_factory=list)  # List of (x, y) wall positions
    direction: str = "right"  # Current direction: "up", "down", "left", "right"
    score: int = 0  # Current score
    game_over: bool = False  # Whether the game has ended
    width: int = 20  # Board width
    height: int = 20  # Board height

    def to_dict(self) -> dict:
        """Convert GameState to a dictionary for JSON serialization."""
        return {
            "snake": [{"x": x, "y": y} for x, y in self.snake],
            "food": {"x": self.food[0], "y": self.food[1]},
            "food_type": self.food_type,
            "food_points": self.food_points,
            "walls": [{"x": x, "y": y} for x, y in self.walls],
            "direction": self.direction,
            "score": self.score,
            "game_over": self.game_over,
            "width": self.width,
            "height": self.height,
        }
