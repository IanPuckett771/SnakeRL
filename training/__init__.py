"""Training infrastructure."""

from training.metrics import MetricsCollector
from training.trainer import Trainer

__all__ = ["Trainer", "MetricsCollector"]
