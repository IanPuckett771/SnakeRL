"""Core game logic for Tank Battle arena combat with smooth movement."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


class Action:
    """Tank actions."""

    FORWARD = 0
    BACKWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    SHOOT = 4


# Physics constants
TANK_SPEED = 2.0  # Pixels per frame
TANK_ROTATION_SPEED = 0.1  # Radians per frame
BULLET_SPEED = 5.0  # Pixels per frame
TANK_RADIUS = 12.0  # Collision radius
BULLET_RADIUS = 3.0
WALL_SIZE = 30.0  # Wall block size
SHOOT_COOLDOWN = 20  # Frames between shots


@dataclass
class Tank:
    """Represents a tank with smooth movement."""

    x: float
    y: float
    angle: float  # Radians, 0 = right, pi/2 = down
    alive: bool = True
    cooldown: int = 0
    health: int = 3

    def copy(self) -> Tank:
        return Tank(
            x=self.x,
            y=self.y,
            angle=self.angle,
            alive=self.alive,
            cooldown=self.cooldown,
            health=self.health,
        )


@dataclass
class Bullet:
    """Represents a bullet with smooth movement."""

    x: float
    y: float
    angle: float
    owner: int  # 0 = player, 1+ = enemy index

    def copy(self) -> Bullet:
        return Bullet(x=self.x, y=self.y, angle=self.angle, owner=self.owner)


@dataclass
class Wall:
    """Rectangular wall obstacle."""

    x: float
    y: float
    width: float
    height: float

    def copy(self) -> Wall:
        return Wall(x=self.x, y=self.y, width=self.width, height=self.height)


@dataclass
class Collectible:
    """Power-up collectible."""

    x: float
    y: float
    type: str  # "health", "speed", "ammo"

    def copy(self) -> Collectible:
        return Collectible(x=self.x, y=self.y, type=self.type)


@dataclass
class TankState:
    """Tank Battle game state with smooth physics."""

    player: Tank
    enemies: list[Tank]
    bullets: list[Bullet]
    walls: list[Wall]
    collectibles: list[Collectible]
    width: float  # Arena width in pixels
    height: float  # Arena height in pixels
    score: int = 0
    game_over: bool = False
    turn: int = 0
    enemies_destroyed: int = 0

    @classmethod
    def create(
        cls,
        width: int = 20,
        height: int = 20,
        num_enemies: int = 3,
    ) -> TankState:
        """Create a new tank battle arena.

        Width/height are in grid units, converted to pixels (20 pixels per unit).
        """
        pixel_width = width * 20.0
        pixel_height = height * 20.0

        # Player starts bottom-left corner (safe from walls)
        player = Tank(
            x=40.0,
            y=pixel_height - 40.0,
            angle=-math.pi / 2,  # Facing up
        )

        # Create enemies in different positions
        enemies = []
        enemy_positions = [
            (pixel_width - 60, 60, math.pi / 2),  # Top-right, facing down
            (pixel_width / 2, 60, math.pi / 2),  # Top-center
            (pixel_width - 60, pixel_height / 2, math.pi),  # Right-center, facing left
        ]
        for i in range(min(num_enemies, len(enemy_positions))):
            ex, ey, ea = enemy_positions[i]
            enemies.append(Tank(x=ex, y=ey, angle=ea))

        # Create some walls
        walls = cls._create_walls(pixel_width, pixel_height)

        # Create initial collectibles
        collectibles = [
            Collectible(x=pixel_width / 2, y=pixel_height / 2, type="health"),
        ]

        return cls(
            player=player,
            enemies=enemies,
            bullets=[],
            walls=walls,
            collectibles=collectibles,
            width=pixel_width,
            height=pixel_height,
        )

    @staticmethod
    def _create_walls(width: float, height: float) -> list[Wall]:
        """Create wall obstacles for the arena."""
        walls = []

        # Some scattered walls (positioned to not block spawn points)
        wall_positions = [
            (width * 0.25, height * 0.3, 60, 20),  # Upper-left area
            (width * 0.7, height * 0.25, 20, 80),  # Upper-right area
            (width * 0.4, height * 0.5, 80, 20),  # Center
            (width * 0.2, height * 0.55, 20, 60),  # Left-center (moved from bottom-left)
            (width * 0.75, height * 0.6, 60, 20),  # Right-center
        ]

        for wx, wy, ww, wh in wall_positions:
            walls.append(Wall(x=wx, y=wy, width=ww, height=wh))

        return walls

    def step(self, action: int) -> tuple[TankState, float, bool]:
        """Execute one game step with smooth physics."""
        self.turn += 1
        reward = -0.01  # Small step penalty

        # Update player cooldown
        if self.player.cooldown > 0:
            self.player.cooldown -= 1

        # Execute player action
        if action == Action.FORWARD:
            self._move_tank(self.player, TANK_SPEED)
        elif action == Action.BACKWARD:
            self._move_tank(self.player, -TANK_SPEED * 0.5)
        elif action == Action.TURN_LEFT:
            self.player.angle -= TANK_ROTATION_SPEED
        elif action == Action.TURN_RIGHT:
            self.player.angle += TANK_ROTATION_SPEED
        elif action == Action.SHOOT:
            if self.player.cooldown <= 0:
                self._fire_bullet(self.player, 0)
                self.player.cooldown = SHOOT_COOLDOWN

        # Update enemies (simple AI)
        for i, enemy in enumerate(self.enemies):
            if not enemy.alive:
                continue
            if enemy.cooldown > 0:
                enemy.cooldown -= 1
            self._update_enemy_ai(enemy, i + 1)

        # Update bullets
        self._update_bullets()

        # Check collisions
        collision_reward = self._check_collisions()
        reward += collision_reward

        # Check collectibles
        collect_reward = self._check_collectibles()
        reward += collect_reward

        # Check win/lose
        if self.player.health <= 0:
            self.player.alive = False
            self.game_over = True
            reward -= 10.0

        if all(not e.alive for e in self.enemies):
            self.game_over = True
            reward += 20.0  # Win bonus

        return self, reward, self.game_over

    def step_multi(
        self,
        forward: bool = False,
        backward: bool = False,
        turn_left: bool = False,
        turn_right: bool = False,
        shoot: bool = False,
    ) -> tuple[TankState, float, bool]:
        """Execute one game step with multiple simultaneous inputs.

        Allows player to move and turn at the same time for smooth controls.

        Args:
            forward: Move forward
            backward: Move backward
            turn_left: Rotate counter-clockwise
            turn_right: Rotate clockwise
            shoot: Fire a bullet

        Returns:
            Updated state, reward, and done flag
        """
        self.turn += 1
        reward = -0.01  # Small step penalty

        # Update player cooldown
        if self.player.cooldown > 0:
            self.player.cooldown -= 1

        # Apply rotation first (so movement is in new direction)
        if turn_left and not turn_right:
            self.player.angle -= TANK_ROTATION_SPEED
        elif turn_right and not turn_left:
            self.player.angle += TANK_ROTATION_SPEED

        # Apply movement
        if forward and not backward:
            self._move_tank(self.player, TANK_SPEED)
        elif backward and not forward:
            self._move_tank(self.player, -TANK_SPEED * 0.5)

        # Handle shooting
        if shoot and self.player.cooldown <= 0:
            self._fire_bullet(self.player, 0)
            self.player.cooldown = SHOOT_COOLDOWN

        # Update enemies (simple AI)
        for i, enemy in enumerate(self.enemies):
            if not enemy.alive:
                continue
            if enemy.cooldown > 0:
                enemy.cooldown -= 1
            self._update_enemy_ai(enemy, i + 1)

        # Update bullets
        self._update_bullets()

        # Check collisions
        collision_reward = self._check_collisions()
        reward += collision_reward

        # Check collectibles
        collect_reward = self._check_collectibles()
        reward += collect_reward

        # Check win/lose
        if self.player.health <= 0:
            self.player.alive = False
            self.game_over = True
            reward -= 10.0

        if all(not e.alive for e in self.enemies):
            self.game_over = True
            reward += 20.0  # Win bonus

        return self, reward, self.game_over

    def _move_tank(self, tank: Tank, speed: float) -> None:
        """Move tank in its facing direction with collision detection."""
        dx = math.cos(tank.angle) * speed
        dy = math.sin(tank.angle) * speed

        new_x = tank.x + dx
        new_y = tank.y + dy

        # Check arena bounds
        new_x = max(TANK_RADIUS, min(self.width - TANK_RADIUS, new_x))
        new_y = max(TANK_RADIUS, min(self.height - TANK_RADIUS, new_y))

        # Check wall collisions
        if not self._collides_with_walls(new_x, new_y, TANK_RADIUS):
            tank.x = new_x
            tank.y = new_y

    def _collides_with_walls(self, x: float, y: float, radius: float) -> bool:
        """Check if a circle collides with any wall."""
        for wall in self.walls:
            # Find closest point on rectangle to circle center
            closest_x = max(wall.x, min(x, wall.x + wall.width))
            closest_y = max(wall.y, min(y, wall.y + wall.height))

            dist = math.sqrt((x - closest_x) ** 2 + (y - closest_y) ** 2)
            if dist < radius:
                return True
        return False

    def _fire_bullet(self, tank: Tank, owner: int) -> None:
        """Fire a bullet from a tank."""
        # Spawn bullet at tank's front
        bx = tank.x + math.cos(tank.angle) * (TANK_RADIUS + BULLET_RADIUS + 2)
        by = tank.y + math.sin(tank.angle) * (TANK_RADIUS + BULLET_RADIUS + 2)
        self.bullets.append(Bullet(x=bx, y=by, angle=tank.angle, owner=owner))

    def _update_enemy_ai(self, enemy: Tank, enemy_idx: int) -> None:
        """Simple enemy AI - turn towards player and shoot."""
        if not self.player.alive:
            return

        # Calculate angle to player
        dx = self.player.x - enemy.x
        dy = self.player.y - enemy.y
        target_angle = math.atan2(dy, dx)

        # Normalize angle difference
        angle_diff = target_angle - enemy.angle
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # Turn towards player
        if abs(angle_diff) > TANK_ROTATION_SPEED:
            if angle_diff > 0:
                enemy.angle += TANK_ROTATION_SPEED * 0.7
            else:
                enemy.angle -= TANK_ROTATION_SPEED * 0.7

        # Move towards player if not too close
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 100:
            self._move_tank(enemy, TANK_SPEED * 0.6)

        # Shoot if roughly facing player
        if abs(angle_diff) < 0.3 and enemy.cooldown <= 0 and random.random() < 0.05:
            self._fire_bullet(enemy, enemy_idx)
            enemy.cooldown = SHOOT_COOLDOWN * 2

    def _update_bullets(self) -> None:
        """Move bullets and remove out-of-bounds ones."""
        new_bullets = []
        for bullet in self.bullets:
            bullet.x += math.cos(bullet.angle) * BULLET_SPEED
            bullet.y += math.sin(bullet.angle) * BULLET_SPEED

            # Check bounds
            if 0 <= bullet.x <= self.width and 0 <= bullet.y <= self.height:
                # Check wall collision
                if not self._collides_with_walls(bullet.x, bullet.y, BULLET_RADIUS):
                    new_bullets.append(bullet)

        self.bullets = new_bullets

    def _check_collisions(self) -> float:
        """Check bullet-tank collisions. Returns reward."""
        reward = 0.0
        remaining_bullets = []

        for bullet in self.bullets:
            hit = False

            # Check player hit (by enemy bullets)
            if bullet.owner != 0 and self.player.alive:
                dist = math.sqrt((bullet.x - self.player.x) ** 2 + (bullet.y - self.player.y) ** 2)
                if dist < TANK_RADIUS + BULLET_RADIUS:
                    self.player.health -= 1
                    hit = True
                    reward -= 2.0

            # Check enemy hits (by player bullets)
            if bullet.owner == 0:
                for i, enemy in enumerate(self.enemies):
                    if not enemy.alive:
                        continue
                    dist = math.sqrt((bullet.x - enemy.x) ** 2 + (bullet.y - enemy.y) ** 2)
                    if dist < TANK_RADIUS + BULLET_RADIUS:
                        enemy.health -= 1
                        if enemy.health <= 0:
                            enemy.alive = False
                            self.enemies_destroyed += 1
                            self.score += 100
                            reward += 10.0
                        hit = True
                        break

            if not hit:
                remaining_bullets.append(bullet)

        self.bullets = remaining_bullets
        return reward

    def _check_collectibles(self) -> float:
        """Check player collecting power-ups."""
        reward = 0.0
        remaining = []

        for coll in self.collectibles:
            dist = math.sqrt((coll.x - self.player.x) ** 2 + (coll.y - self.player.y) ** 2)
            if dist < TANK_RADIUS + 10:
                if coll.type == "health":
                    self.player.health = min(5, self.player.health + 1)
                    reward += 5.0
                    self.score += 25
            else:
                remaining.append(coll)

        self.collectibles = remaining

        # Spawn new collectible occasionally
        if len(self.collectibles) < 2 and random.random() < 0.01:
            cx = random.uniform(50, self.width - 50)
            cy = random.uniform(50, self.height - 50)
            self.collectibles.append(Collectible(x=cx, y=cy, type="health"))

        return reward

    def reset(self) -> TankState:
        """Reset the game to initial state."""
        # Get grid dimensions from current pixel size
        grid_width = int(self.width / 20)
        grid_height = int(self.height / 20)
        return TankState.create(
            width=grid_width,
            height=grid_height,
            num_enemies=len(self.enemies),
        )

    def copy(self) -> TankState:
        """Create a deep copy of this state."""
        return TankState(
            player=self.player.copy(),
            enemies=[e.copy() for e in self.enemies],
            bullets=[b.copy() for b in self.bullets],
            walls=[w.copy() for w in self.walls],
            collectibles=[c.copy() for c in self.collectibles],
            width=self.width,
            height=self.height,
            score=self.score,
            game_over=self.game_over,
            turn=self.turn,
            enemies_destroyed=self.enemies_destroyed,
        )

    def get_score(self) -> int:
        """Get current score."""
        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "player": {
                "x": self.player.x,
                "y": self.player.y,
                "angle": self.player.angle,
                "alive": self.player.alive,
                "health": self.player.health,
            },
            "enemies": [
                {
                    "x": e.x,
                    "y": e.y,
                    "angle": e.angle,
                    "alive": e.alive,
                    "health": e.health,
                }
                for e in self.enemies
                if e.alive
            ],
            "bullets": [
                {"x": b.x, "y": b.y, "angle": b.angle, "owner": b.owner} for b in self.bullets
            ],
            "walls": [
                {"x": w.x, "y": w.y, "width": w.width, "height": w.height} for w in self.walls
            ],
            "collectibles": [{"x": c.x, "y": c.y, "type": c.type} for c in self.collectibles],
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "health": self.player.health,
            "game_over": self.game_over,
            "turn": self.turn,
            "enemies_destroyed": self.enemies_destroyed,
        }
