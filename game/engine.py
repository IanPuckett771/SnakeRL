import random
from typing import Tuple, Optional

from .state import GameState


class SnakeGame:
    """Snake game engine with configurable board size."""

    DIRECTIONS = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    OPPOSITE_DIRECTIONS = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }

    def __init__(self, width: int = 20, height: int = 20):
        """Initialize the game with configurable board size.

        Args:
            width: Board width in cells
            height: Board height in cells
        """
        self.width = width
        self.height = height
        self.snake: list[Tuple[int, int]] = []
        self.food: Tuple[int, int] = (0, 0)
        self.direction: str = "right"
        self.score: int = 0
        self.game_over: bool = False
        self.reset()

    def reset(self) -> GameState:
        """Reset the game to initial state.

        Snake starts in the center, food is placed randomly.

        Returns:
            The initial GameState
        """
        # Snake starts in center, length 3, facing right
        center_x = self.width // 2
        center_y = self.height // 2

        self.snake = [
            (center_x, center_y),      # Head
            (center_x - 1, center_y),  # Body
            (center_x - 2, center_y),  # Tail
        ]

        self.direction = "right"
        self.score = 0
        self.game_over = False

        # Place food in random empty cell
        self._spawn_food()

        return self.get_state()

    def _spawn_food(self) -> None:
        """Spawn food in a random empty cell."""
        # Get all empty cells
        snake_set = set(self.snake)
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in snake_set
        ]

        if empty_cells:
            self.food = random.choice(empty_cells)
        else:
            # No empty cells (snake fills board - win condition)
            self.food = (-1, -1)

    def _check_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with walls or snake body.

        Args:
            position: (x, y) position to check

        Returns:
            True if collision detected, False otherwise
        """
        x, y = position

        # Wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Self collision (check against body, not including tail since it will move)
        if position in self.snake[:-1]:
            return True

        return False

    def step(self, action: Optional[str] = None) -> Tuple[GameState, float, bool]:
        """Process one game step with the given action.

        Args:
            action: Direction to move ("up", "down", "left", "right").
                   If None or invalid, continues in current direction.
                   Cannot reverse direction (ignored if attempted).

        Returns:
            Tuple of (GameState, reward, done)
            - reward: +10 for eating food, -10 for death, 0 otherwise
            - done: True if game is over
        """
        if self.game_over:
            return self.get_state(), 0.0, True

        # Update direction if valid action provided
        if action in self.DIRECTIONS:
            # Prevent reversing direction (can't go back on yourself)
            if action != self.OPPOSITE_DIRECTIONS.get(self.direction):
                self.direction = action

        # Calculate new head position
        dx, dy = self.DIRECTIONS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check for collision
        if self._check_collision(new_head):
            self.game_over = True
            return self.get_state(), -10.0, True

        # Move snake
        self.snake.insert(0, new_head)

        # Check if food eaten
        reward = 0.0
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self._spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()

        return self.get_state(), reward, False

    def get_state(self) -> GameState:
        """Get the current game state.

        Returns:
            Current GameState
        """
        return GameState(
            snake=list(self.snake),
            food=self.food,
            direction=self.direction,
            score=self.score,
            game_over=self.game_over,
            width=self.width,
            height=self.height,
        )
