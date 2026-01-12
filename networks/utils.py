"""Network utilities including device detection."""

from __future__ import annotations

import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_gpu: If True, prefer GPU over CPU if available

    Returns:
        torch.device for computation
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_gpu and torch.backends.mps.is_available():
        # Apple Silicon GPU
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """Get information about available devices."""
    info: dict[str, object] = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info
