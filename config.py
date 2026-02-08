"""Global configuration for SnakeRL training optimization.

This module provides centralized control over data types and memory optimization
settings to speed up training.
"""

import numpy as np

# ============================================================================
# DATA TYPE CONFIGURATION FOR MEMORY & SPEED OPTIMIZATION
# ============================================================================

# Observation dtype: Controls memory usage and computation speed
# Options:
#   - np.float32: Standard precision (baseline)
#   - np.float16: Half precision (50% memory reduction, faster on modern GPUs)
#   - np.uint8:   8-bit integers (75% memory reduction, best for CPU memory)
#
# Recommendation:
#   - Use float16 for GPU training (best balance of speed/memory/accuracy)
#   - Use uint8 if memory-constrained or training on CPU
#   - Use float32 if experiencing numerical stability issues
OBSERVATION_DTYPE = np.float16

# NOTE: Mixed precision training (torch.cuda.amp) is not yet implemented.
# This flag is reserved for future use.
USE_MIXED_PRECISION = False

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================
# Expected improvements with float16 over float32:
#   - Memory usage: ~50% reduction in replay buffer and observation storage
#   - GPU throughput: 1.5-2x faster on GPUs with Tensor Cores (RTX 20xx+, V100+)
#   - Batch processing: Can train with larger batch sizes due to memory savings
#   - Numerical precision: Sufficient for RL (values are mostly binary 0/1)
#
# Expected improvements with uint8 over float32:
#   - Memory usage: ~75% reduction
#   - CPU memory pressure: Significantly reduced
#   - Transfer overhead: Reduced CPU->GPU transfer time
#   - Trade-off: Requires conversion to float in data pipeline


def get_observation_dtype() -> np.dtype:
    """Get the configured observation data type.
    
    Returns:
        numpy dtype for observations
    """
    return OBSERVATION_DTYPE


def dtype_name() -> str:
    """Get human-readable name of current dtype.
    
    Returns:
        String name of dtype (e.g., 'float16', 'float32', 'uint8')
    """
    return OBSERVATION_DTYPE.__name__


def memory_multiplier() -> float:
    """Get memory usage multiplier relative to float32 baseline.
    
    Returns:
        Memory multiplier (e.g., 0.5 for float16, 0.25 for uint8)
    """
    dtype_sizes = {
        np.float32: 1.0,
        np.float16: 0.5,
        np.uint8: 0.25,
    }
    return dtype_sizes.get(OBSERVATION_DTYPE, 1.0)


def print_config() -> None:
    """Print current configuration settings."""
    print("=" * 70)
    print("SnakeRL Configuration")
    print("=" * 70)
    print(f"Observation dtype: {dtype_name()}")
    print(f"Memory usage: {memory_multiplier():.0%} of float32 baseline")
    print(f"Mixed precision training: {USE_MIXED_PRECISION}")
    
    if OBSERVATION_DTYPE == np.float16:
        print("\n[OK] Using float16 for observations (50% memory reduction)")
        print("  -> Best for GPU training with modern hardware")
    elif OBSERVATION_DTYPE == np.uint8:
        print("\n[OK] Using uint8 for observations (75% memory reduction)")
        print("  -> Best for CPU training or extreme memory constraints")
        print("  [!] Requires conversion to float in training pipeline")
    else:
        print("\n  Using float32 baseline (no optimization)")
    
    print("=" * 70)


if __name__ == "__main__":
    print_config()
