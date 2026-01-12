"""Metrics collection and logging."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class MetricsCollector:
    """Collect and aggregate training metrics."""

    def __init__(self):
        self.metrics: dict[str, list[tuple]] = defaultdict(list)
        self._step = 0

    def add(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Add metrics at a given step.

        Args:
            metrics: Dict of metric name -> value
            step: Optional step number (uses internal counter if None)
        """
        if step is None:
            step = self._step

        for name, value in metrics.items():
            self.metrics[name].append((step, value))

        self._step = step + 1

    def get(self, name: str) -> list[tuple]:
        """Get all values for a metric."""
        return self.metrics.get(name, [])

    def get_latest(self, name: str) -> float | None:
        """Get the latest value for a metric."""
        values = self.metrics.get(name, [])
        if values:
            return float(values[-1][1])
        return None

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics for all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                vals = [v for _, v in values]
                summary[name] = {
                    "last": vals[-1],
                    "mean": sum(vals) / len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "count": len(vals),
                }
        return summary

    def save(self, path: str) -> None:
        """Save metrics to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            name: [(step, float(val)) for step, val in values]
            for name, values in self.metrics.items()
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load metrics from JSON file."""
        with open(path) as f:
            data = json.load(f)

        self.metrics = defaultdict(list)
        for name, values in data.items():
            self.metrics[name] = [(step, val) for step, val in values]
