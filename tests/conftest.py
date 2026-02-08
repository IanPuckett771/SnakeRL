"""Pytest configuration and fixtures."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get test device (CPU for consistency)."""
    return torch.device("cpu")


@pytest.fixture
def observation_shape() -> tuple[int, int, int]:
    """Standard observation shape for tests."""
    return (3, 20, 20)


@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(seed: int) -> None:
    """Set random seeds for all tests."""
    np.random.seed(seed)
    torch.manual_seed(seed)
