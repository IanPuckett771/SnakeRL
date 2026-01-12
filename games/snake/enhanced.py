"""Enhanced Snake environment with obstacles and multiple food types."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from games.base import BaseGameEnv, GameMetadata
from games.registry import GameRegistry


class FoodType(Enum):
    """Different types of food with varying effects."""

    REGULAR = "regular"  # +10 points, grow by 1
    SUPER = "super"  # +25 points, grow by 1
    SPEED = "speed"  # +5 points, temporary speed boost (handled by frontend)
    SHRINK = "shrink"  # +15 points, snake shrinks by 1


@dataclass
class Food:
    """A food item on the board."""

    x: int
    y: int
    food_type: FoodType

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "type": self.food_type.value,
        }


@dataclass
class Obstacle:
    """An obstacle on the board."""

    x: int
    y: int

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y}


class EnhancedSnakeEnv(BaseGameEnv):
    """Enhanced Snake with obstacles and multiple food types.

    Features:
    - Static obstacles that kill the snake on collision
    - Multiple food types with different rewards:
        - Regular (green): +10 points, grow by 1
        - Super (gold): +25 points, grow by 1
        - Speed (blue): +5 points, temporary speed indicator
        - Shrink (red): +15 points, snake shrinks by 1
    """

    ACTION_MAP = {0: "up", 1: "down", 2: "left", 3: "right"}
    ACTION_NAMES = ["up", "down", "left", "right"]

    DIRECTIONS = {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }

    OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

    # Food spawn probabilities
    FOOD_WEIGHTS = {
        FoodType.REGULAR: 0.5,
        FoodType.SUPER: 0.15,
        FoodType.SPEED: 0.2,
        FoodType.SHRINK: 0.15,
    }

    # Reward values for each food type
    FOOD_REWARDS = {
        FoodType.REGULAR: 10.0,
        FoodType.SUPER: 25.0,
        FoodType.SPEED: 5.0,
        FoodType.SHRINK: 15.0,
    }

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        num_obstacles: int = 5,
        num_food: int = 3,
    ):
        """Initialize enhanced snake environment.

        Args:
            width: Board width
            height: Board height
            num_obstacles: Number of static obstacles
            num_food: Number of food items on board at once
        """
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.num_food = num_food

        self._max_steps = width * height * 10

        # Observation: 6 channels
        # 0: snake body
        # 1: snake head
        # 2: regular food
        # 3: super food
        # 4: speed food / shrink food (combined for simplicity)
        # 5: obstacles
        self.metadata = GameMetadata(
            name="enhanced_snake",
            action_space_size=4,
            action_names=self.ACTION_NAMES,
            observation_shape=(6, height, width),
            max_episode_steps=self._max_steps,
            supports_mcts=True,
        )

        # Game state
        self.snake: list[tuple[int, int]] = []
        self.direction: str = "right"
        self.foods: list[Food] = []
        self.obstacles: list[Obstacle] = []
        self.score: int = 0
        self.game_over: bool = False
        self._step_count: int = 0

        self.reset()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the game."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset state
        self.score = 0
        self.game_over = False
        self._step_count = 0
        self.direction = "right"

        # Initialize snake in center
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]

        # Place obstacles (not on snake starting area)
        self.obstacles = []
        self._spawn_obstacles()

        # Place food
        self.foods = []
        for _ in range(self.num_food):
            self._spawn_food()

        obs = self._get_observation()
        info = {"score": 0}
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step."""
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {"score": self.score}

        # Update direction
        action_str = self.ACTION_MAP.get(action, "right")
        if action_str != self.OPPOSITE.get(self.direction):
            self.direction = action_str

        # Calculate new head position
        dx, dy = self.DIRECTIONS[self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Check collisions
        reward = 0.0

        # Wall collision
        if (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
        ):
            self.game_over = True
            return self._get_observation(), -10.0, True, False, {"score": self.score}

        # Self collision
        if new_head in self.snake[:-1]:
            self.game_over = True
            return self._get_observation(), -10.0, True, False, {"score": self.score}

        # Obstacle collision
        for obstacle in self.obstacles:
            if new_head == (obstacle.x, obstacle.y):
                self.game_over = True
                return self._get_observation(), -10.0, True, False, {"score": self.score}

        # Move snake
        self.snake.insert(0, new_head)

        # Check for food
        ate_food = None
        for food in self.foods:
            if new_head == (food.x, food.y):
                ate_food = food
                break

        if ate_food:
            # Get reward
            reward = self.FOOD_REWARDS[ate_food.food_type]
            self.score += int(reward)

            # Handle food type effects
            if ate_food.food_type == FoodType.SHRINK:
                # Shrink snake (remove 2 segments if possible, net -1)
                if len(self.snake) > 2:
                    self.snake.pop()
                    self.snake.pop()
            elif ate_food.food_type in (FoodType.REGULAR, FoodType.SUPER):
                # Grow - don't remove tail
                pass
            else:  # SPEED
                # Don't grow, remove tail
                self.snake.pop()

            # Remove eaten food and spawn new one
            self.foods.remove(ate_food)
            self._spawn_food()
        else:
            # Remove tail (no food eaten)
            self.snake.pop()

        self._step_count += 1
        truncated = self._step_count >= self._max_steps

        obs = self._get_observation()
        info = {"score": self.score, "length": len(self.snake)}

        return obs, reward, self.game_over, truncated, info

    def get_valid_actions(self) -> list[int]:
        """All 4 directions are technically valid."""
        return [0, 1, 2, 3]

    def clone(self) -> EnhancedSnakeEnv:
        """Create deep copy for MCTS simulation."""
        new_env = EnhancedSnakeEnv(
            width=self.width,
            height=self.height,
            num_obstacles=self.num_obstacles,
            num_food=self.num_food,
        )
        new_env.snake = list(self.snake)
        new_env.direction = self.direction
        new_env.foods = [Food(f.x, f.y, f.food_type) for f in self.foods]
        new_env.obstacles = [Obstacle(o.x, o.y) for o in self.obstacles]
        new_env.score = self.score
        new_env.game_over = self.game_over
        new_env._step_count = self._step_count
        return new_env

    def render_state(self) -> dict[str, Any]:
        """Return state for WebSocket rendering."""
        return {
            "snake": [{"x": x, "y": y} for x, y in self.snake],
            "foods": [f.to_dict() for f in self.foods],
            "obstacles": [o.to_dict() for o in self.obstacles],
            "direction": self.direction,
            "score": self.score,
            "game_over": self.game_over,
            "width": self.width,
            "height": self.height,
            "enhanced": True,
        }

    def get_observation(self) -> np.ndarray:
        """Get current observation."""
        return self._get_observation()

    def _get_observation(self) -> np.ndarray:
        """Convert game state to neural network input.

        Returns 6-channel grid:
        - Channel 0: Snake body
        - Channel 1: Snake head
        - Channel 2: Regular food
        - Channel 3: Super food
        - Channel 4: Speed/Shrink food
        - Channel 5: Obstacles
        """
        obs = np.zeros((6, self.height, self.width), dtype=np.float32)

        # Snake body (excluding head)
        for x, y in self.snake[1:]:
            if 0 <= x < self.width and 0 <= y < self.height:
                obs[0, y, x] = 1.0

        # Snake head
        if self.snake:
            hx, hy = self.snake[0]
            if 0 <= hx < self.width and 0 <= hy < self.height:
                obs[1, hy, hx] = 1.0

        # Foods
        for food in self.foods:
            if 0 <= food.x < self.width and 0 <= food.y < self.height:
                if food.food_type == FoodType.REGULAR:
                    obs[2, food.y, food.x] = 1.0
                elif food.food_type == FoodType.SUPER:
                    obs[3, food.y, food.x] = 1.0
                else:  # SPEED or SHRINK
                    obs[4, food.y, food.x] = 1.0

        # Obstacles
        for obstacle in self.obstacles:
            if 0 <= obstacle.x < self.width and 0 <= obstacle.y < self.height:
                obs[5, obstacle.y, obstacle.x] = 1.0

        return obs

    def _spawn_obstacles(self) -> None:
        """Spawn obstacles avoiding snake starting area."""
        # Define safe zone around snake start
        center_x = self.width // 2
        center_y = self.height // 2
        safe_zone = set()
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                safe_zone.add((center_x + dx, center_y + dy))

        attempts = 0
        while len(self.obstacles) < self.num_obstacles and attempts < 100:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)

            if (x, y) not in safe_zone:
                # Check not on existing obstacle
                if not any(o.x == x and o.y == y for o in self.obstacles):
                    self.obstacles.append(Obstacle(x, y))

            attempts += 1

    def _spawn_food(self) -> None:
        """Spawn a new food item."""
        # Get occupied positions
        occupied = set(self.snake)
        occupied.update((o.x, o.y) for o in self.obstacles)
        occupied.update((f.x, f.y) for f in self.foods)

        # Find empty positions
        empty = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in occupied:
                    empty.append((x, y))

        if not empty:
            return

        # Choose random position
        x, y = random.choice(empty)

        # Choose food type based on weights
        types = list(self.FOOD_WEIGHTS.keys())
        weights = list(self.FOOD_WEIGHTS.values())
        food_type = random.choices(types, weights=weights, k=1)[0]

        self.foods.append(Food(x, y, food_type))


# Register the enhanced game
GameRegistry.register("enhanced_snake", EnhancedSnakeEnv)
