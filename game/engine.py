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

    # Treat types: (color, points, spawn_weight)
    # Lower weight = less likely to spawn
    TREAT_TYPES = [
        ("red", 1, 50),      # Common - red, 1 point
        ("orange", 2, 30),   # Uncommon - orange, 2 points
        ("yellow", 5, 15),   # Rare - yellow, 5 points
        ("green", 10, 4),    # Very rare - green, 10 points
        ("blue", 20, 1),    # Ultra rare - blue, 20 points
    ]

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
        self.food_type: str = "red"  # Color/type of current food
        self.food_points: int = 1    # Points this food is worth
        self.walls: list[Tuple[int, int]] = []  # List of wall positions
        self.direction: str = "right"
        self.score: int = 0
        self.steps_since_last_treat: int = 0  # Track steps without eating a treat
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
        self.walls = []
        self.steps_since_last_treat = 0

        # Place food in random empty cell
        self._spawn_food()
        
        # Walls disabled - focus on filling the board first
        # self._spawn_walls()

        return self.get_state()

    # Pre-compute treat weights for random.choices (avoid recomputing each spawn)
    _TREAT_WEIGHTS = [treat[2] for treat in TREAT_TYPES]

    def _spawn_food(self) -> None:
        """Spawn food in a random empty cell with weighted random type selection.
        Uses rejection sampling for speed when board is mostly empty."""
        snake_set = set(self.snake)
        wall_set = set(self.walls)
        occupied_count = len(snake_set) + len(wall_set)
        total_cells = self.width * self.height
        
        # Fast path: if board is <50% full, use rejection sampling (much faster than scanning)
        if occupied_count < total_cells * 0.5:
            blocked = snake_set | wall_set
            for _ in range(100):  # Max attempts
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if (x, y) not in blocked:
                    treat_type = random.choices(self.TREAT_TYPES, weights=self._TREAT_WEIGHTS, k=1)[0]
                    self.food = (x, y)
                    self.food_type = treat_type[0]
                    self.food_points = treat_type[1]
                    return
        
        # Slow path: enumerate all empty cells (when board is crowded)
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in snake_set and (x, y) not in wall_set
        ]

        if empty_cells:
            treat_type = random.choices(self.TREAT_TYPES, weights=self._TREAT_WEIGHTS, k=1)[0]
            self.food = random.choice(empty_cells)
            self.food_type = treat_type[0]
            self.food_points = treat_type[1]
        else:
            self.food = (-1, -1)
            self.food_type = "red"
            self.food_points = 1

    def _spawn_walls(self) -> None:
        """Spawn random walls until the player eats a treat.
        Walls are placed in empty cells, avoiding snake and food.
        Wall count scales with snake length - easier early, harder later."""
        # Clear existing walls
        self.walls = []
        
        # Scale wall count with snake length - no walls early, more walls as snake grows
        snake_len = len(self.snake)
        if snake_len < 5:
            # No walls until snake reaches length 5 - learn basics first
            return
        
        # Get all occupied cells (snake + food)
        occupied = set(self.snake)
        if self.food[0] >= 0 and self.food[1] >= 0:
            occupied.add(self.food)
        
        # Get all empty cells
        empty_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        
        if not empty_cells:
            return
        
        # Scale walls with snake length: 1 wall per 5 length, max 8 walls
        base_walls = (snake_len - 5) // 5 + 1  # 1 wall at length 5, 2 at 10, etc.
        num_walls = min(base_walls, 8, len(empty_cells) // 4)
        
        if num_walls <= 0:
            return
        
        # Randomly select wall positions
        wall_positions = random.sample(empty_cells, min(num_walls, len(empty_cells)))
        self.walls = wall_positions

    def _check_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position collides with walls or snake body.

        Args:
            position: (x, y) position to check

        Returns:
            True if collision detected, False otherwise
        """
        x, y = position

        # Boundary wall collision
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        # Game wall collision
        if position in self.walls:
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
            - reward: Based on speed of treat collection (faster = higher reward)
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
            # HARSH death penalty - scales significantly with snake length
            # Dying with a long snake should be very costly to discourage risky behavior
            snake_len = len(self.snake)
            death_penalty = -100.0 - (snake_len * 10)  # Length 3 = -130, Length 10 = -200, Length 20 = -300
            return self.get_state(), death_penalty, True

        # Move snake
        old_snake_length = len(self.snake)
        self.snake.insert(0, new_head)
        self.steps_since_last_treat += 1

        # Calculate distance change for direction incentive
        old_head = self.snake[1] if len(self.snake) > 1 else self.snake[0]
        old_distance = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        # REWARD STRUCTURE FOR PERFECT GAME (fill entire 20x20 board)
        # Key changes from previous version:
        # - Time pressure scales DOWN as snake grows (long snakes NEED more time to navigate)
        # - More milestone bonuses up to 400 (full board)
        # - Softer direction incentive so snake can take necessary detours
        reward = 0.0
        
        # Soft direction incentive - allows detours around body
        if new_distance < old_distance:
            reward += 0.1
        elif new_distance > old_distance:
            reward -= 0.15
        
        if new_head == self.food:
            # TREAT COLLECTED! Reward based on speed
            max_reasonable_steps = self.width + self.height
            speed_multiplier = max(0.1, (max_reasonable_steps - self.steps_since_last_treat) / max_reasonable_steps)
            
            # Base reward scaled by treat value and speed
            base_reward = float(self.food_points * 10)
            reward = base_reward * (1.0 + speed_multiplier)
            
            # MILESTONE BONUSES - scaled up dramatically for long snakes
            new_length = len(self.snake)
            milestones = [
                (5, 50), (10, 100), (15, 150), (20, 200),
                (30, 300), (50, 500), (75, 750), (100, 1000),
                (150, 1500), (200, 2000), (250, 3000), (300, 4000),
                (350, 5000), (375, 7500), (390, 10000), (400, 20000),
            ]
            for threshold, bonus in milestones:
                if new_length >= threshold and old_snake_length < threshold:
                    reward += bonus
            
            # Add points to score
            self.score += self.food_points
            
            # Reset counter and spawn new food
            self.steps_since_last_treat = 0
            self._spawn_food()
        else:
            # Remove tail if no food eaten
            self.snake.pop()
            
            # Time pressure SCALES with snake length:
            # Short snake (len 3-20): -0.03 per step (move quickly!)
            # Long snake (len 50+): -0.01 per step (more time to navigate)
            # Very long snake (200+): -0.005 per step (needs lots of time for detours)
            snake_len = len(self.snake)
            if snake_len < 20:
                time_penalty = 0.03
            elif snake_len < 50:
                time_penalty = 0.02
            elif snake_len < 100:
                time_penalty = 0.01
            elif snake_len < 200:
                time_penalty = 0.007
            else:
                time_penalty = 0.005
            reward -= time_penalty

        return self.get_state(), reward, False

    def get_state(self) -> GameState:
        """Get the current game state.

        Returns:
            Current GameState
        """
        return GameState(
            snake=list(self.snake),
            food=self.food,
            food_type=self.food_type,
            food_points=self.food_points,
            walls=list(self.walls),
            direction=self.direction,
            score=self.score,
            game_over=self.game_over,
            width=self.width,
            height=self.height,
        )
