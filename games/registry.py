"""Game registry for dynamic game loading."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from games.base import BaseGameEnv


class GameRegistry:
    """Registry for game environments."""

    _games: dict[str, Callable[..., BaseGameEnv]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., BaseGameEnv]) -> None:
        """Register a game factory function.

        Args:
            name: Unique game identifier
            factory: Function that creates a game instance
        """
        cls._games[name] = factory

    @classmethod
    def create(cls, name: str, config: dict[str, Any] | None = None) -> BaseGameEnv:
        """Create a game instance.

        Args:
            name: Game identifier
            config: Optional configuration dict

        Returns:
            Game environment instance
        """
        if name not in cls._games:
            raise ValueError(f"Unknown game: {name}. Available: {list(cls._games.keys())}")

        config = config or {}
        return cls._games[name](**config)

    @classmethod
    def list_games(cls) -> list[str]:
        """Return list of registered game names."""
        return list(cls._games.keys())

    @classmethod
    def has_game(cls, name: str) -> bool:
        """Check if a game is registered."""
        return name in cls._games
