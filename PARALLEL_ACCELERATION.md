# Parallel Acceleration & GPU Support

This document describes the parallel evaluation and GPU acceleration features added to the Spin Network calculator.

## Overview

The evaluator now supports:
- ✅ **Multi-variable summations** (arbitrary N nested summations)
- ✅ **CPU parallelization** via multiprocessing
- ✅ **GPU acceleration** via JAX (Apple Metal, NVIDIA CUDA, or CPU)
- ✅ **Automatic backend selection** (always uses fastest available)
- ✅ **NumPy vectorization** for theta/delta symbols

## Quick Start

**No code changes required!** Parallel evaluation is now the default:

```python
from src.spin_evaluator import evaluate_spin_network

# Automatically uses best available backend (GPU or multiprocessing)
result = evaluate_spin_network(canonical_terms, max_two_j=200)
```

## Installation

### Core Dependencies (Required)

```bash
pip install numpy>=1.20.0
```

### GPU Acceleration (Optional but Recommended)

**For Apple Silicon (M1/M2/M3):**
```bash
pip install jax-metal
```

**For NVIDIA CUDA GPUs:**
```bash
pip install jax[cuda]
```

**For CPU-only JAX (still faster than pure Python):**
```bash
pip install jax
```

Or install all at once:
```bash
pip install -r requirements.txt
```

## Backend Selection

The evaluator automatically selects the fastest available backend:

1. **JAX GPU** (if GPU detected) - Fastest
2. **JAX CPU** (if JAX installed) - Fast
3. **Multiprocessing** (fallback) - Good speedup
4. **Serial** (only for debugging) - Slowest

### Manual Backend Selection

```python
from src.spin_evaluator import SpinNetworkEvaluator

# Force specific backend
evaluator = SpinNetworkEvaluator(
    max_two_j=200,
    backend='multiprocessing',  # Options: 'auto', 'jax', 'multiprocessing', 'serial'
    n_workers=8                  # Number of CPU workers
)
```

## Multi-Variable Summations

The evaluator now handles **arbitrary N nested summations** automatically:

### Example: Double Summation

```python
term = {
    "coeffs": [
        {"type": "sum", "index": "F_1", "range2": {"Fmin": 2, "Fmax": 10}},
        {"type": "sum", "index": "F_2", "range2": {"Fmin": 2, "Fmax": 10}},
        {"type": "delta", "fixed": {"j": "F_1"}, "power": 1},
        {"type": "delta", "fixed": {"j": "F_2"}, "power": 1},
        {"type": "W6j", "args": (1, "F_1", 2, "F_2", 3, 1), "power": 1}
    ]
}

# Evaluates ∑_{F_1=1..5} ∑_{F_2=1..5} [Δ_{F_1} × Δ_{F_2} × W6j(...)]
result = evaluate_spin_network([term], max_two_j=50)
```

### Example: Triple Summation

```python
term = {
    "coeffs": [
        {"type": "sum", "index": "F_1", "range2": {"Fmin": 2, "Fmax": 8}},
        {"type": "sum", "index": "F_2", "range2": {"Fmin": 2, "Fmax": 8}},
        {"type": "sum", "index": "F_3", "range2": {"Fmin": 2, "Fmax": 8}},
        # ... coefficients using F_1, F_2, F_3 ...
    ]
}

# Automatically parallelized across CPU cores
result = evaluate_spin_network([term])
```

**No limit on the number of summation variables!** The code uses `itertools.product` to generate all combinations and evaluates them in parallel.

## Performance

### Expected Speedups (vs. serial evaluation)

| Backend | Typical Speedup | Hardware Required |
|---------|----------------|-------------------|
| JAX GPU | 10-100x | NVIDIA GPU or Apple M1/M2/M3 |
| JAX CPU | 3-10x | Any CPU |
| Multiprocessing | 2-8x | Multi-core CPU |

### Benchmarking

Compare all available backends on your hardware:

```python
from src.spin_evaluator import benchmark_backends

# Runs evaluation on all backends and reports timings
times = benchmark_backends(canonical_terms, max_two_j=200)

# Output:
# ======================================================================
# BENCHMARK SUMMARY
# ======================================================================
#   jax: 0.845s (speedup: 6.19x)
#   multiprocessing: 1.234s (speedup: 4.24x)
#   serial: 5.234s (speedup: 1.00x)
#
# 🏆 Best backend: jax (0.845s)
# ======================================================================
```

## Implementation Details

### Parallel Strategy

**Single summation variable:**
- Chunks the range into N pieces (N = number of workers)
- Each worker evaluates its chunk independently
- Results are summed at the end

**Multiple summation variables:**
- Uses `itertools.product()` to generate all combinations
- Chunks the full combination space
- Each worker evaluates a chunk in parallel
- No inter-process communication needed (embarrassingly parallel)

### Progress Reporting

For large summations (>1000 iterations), progress is automatically reported:

```
Computing summation over 2 variable(s)...
  Total iterations: 10,000
  Using 7 parallel workers
  Split into 28 chunks of ~357 iterations each
  Progress: 1,000/10,000 (10.0%)
  Progress: 2,000/10,000 (20.0%)
  ...
```

### Limitations

**JAX Backend:**
- The Wigner 6j symbols still use `pywigxjpf` (C++ library, already optimized)
- JAX currently accelerates theta/delta computations
- Full JAX implementation would require JAX-native 6j symbol implementation

**Memory:**
- For very large summation spaces (e.g., 5+ variables with wide ranges), the combination list may consume significant memory
- Consider chunking or using serial backend for extreme cases

## Advanced Usage

### Custom Worker Count

```python
from multiprocessing import cpu_count

# Use all but one CPU core
n_workers = cpu_count() - 1

evaluator = SpinNetworkEvaluator(
    max_two_j=200,
    backend='multiprocessing',
    n_workers=n_workers
)
```

### Disable Parallelization (for debugging)

```python
# Use single-threaded evaluation (slow, but easier to debug)
evaluator = SpinNetworkEvaluator(max_two_j=200, backend='serial')
```

### Checking Available Backends

```python
from src.spin_evaluator import JAX_AVAILABLE, JAX_GPU_AVAILABLE

print(f"JAX installed: {JAX_AVAILABLE}")
print(f"JAX GPU available: {JAX_GPU_AVAILABLE}")

# On Apple M3:
# JAX installed: True
# JAX GPU available: True
```

## Troubleshooting

### "JAX backend requested but JAX not installed"

Install JAX:
```bash
pip install jax  # CPU only
# OR
pip install jax-metal  # Apple Silicon GPU
```

### Slow performance on M3

Make sure you have `jax-metal` installed for GPU acceleration:
```bash
pip uninstall jax  # Remove CPU-only version
pip install jax-metal
```

### Multiprocessing issues on macOS

If you see errors like "RuntimeError: context has already been set", add this to your main script:

```python
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    # ... rest of your code
```

### Out of memory errors

For very large summation spaces, reduce the number of workers or use serial backend:

```python
# Reduce workers to save memory
evaluator = SpinNetworkEvaluator(max_two_j=200, backend='multiprocessing', n_workers=2)

# OR use serial for extreme cases
evaluator = SpinNetworkEvaluator(max_two_j=200, backend='serial')
```

## API Reference

### `SpinNetworkEvaluator`

```python
SpinNetworkEvaluator(max_two_j=200, backend='auto', n_workers=None)
```

**Parameters:**
- `max_two_j` (int): Maximum 2*j value expected
- `backend` (str): 'auto', 'jax', 'multiprocessing', or 'serial'
- `n_workers` (int): Number of parallel workers (defaults to CPU count - 1)

### `evaluate_spin_network`

```python
evaluate_spin_network(canonical_terms, max_two_j=200, backend='auto', n_workers=None)
```

Convenience function that automatically creates evaluator, runs evaluation, and cleans up.

**Parameters:**
- `canonical_terms` (list): Output from `canonicalise_terms()`
- `max_two_j` (int): Maximum 2*j value
- `backend` (str): Backend selection
- `n_workers` (int): Number of workers

**Returns:** float (numerical result)

### `benchmark_backends`

```python
benchmark_backends(canonical_terms, max_two_j=200)
```

Benchmarks all available backends and returns timing dictionary.

**Returns:** dict mapping backend name to execution time

## Migration Guide

**No changes needed!** Existing code automatically benefits from parallelization:

### Before (old code still works):
```python
from src.spin_evaluator import evaluate_spin_network
result = evaluate_spin_network(canon_terms)
```

### After (same code, now runs in parallel):
```python
from src.spin_evaluator import evaluate_spin_network
result = evaluate_spin_network(canon_terms)  # Now uses multiprocessing/GPU automatically!
```

### Optional: Explicit backend control
```python
# Force specific backend if needed
result = evaluate_spin_network(canon_terms, backend='jax')  # Use GPU
result = evaluate_spin_network(canon_terms, backend='multiprocessing', n_workers=4)  # 4 cores
result = evaluate_spin_network(canon_terms, backend='serial')  # Debug mode
```

## Contributing

To add a new backend:

1. Add imports at top of `spin_evaluator.py`
2. Update backend selection logic in `__init__`
3. Implement `_evaluate_sum_BACKEND()` method
4. Update documentation and tests

See existing `_evaluate_sum_jax()` and `_evaluate_sum_parallel()` for examples.
