# Parallel Acceleration & GPU Support

This document describes the parallel evaluation and GPU acceleration features of the Spin Network calculator.

## Overview

The evaluator supports:
- **CPU parallelization** via multiprocessing
- **GPU acceleration** via JAX (Apple Metal, NVIDIA CUDA, or CPU)
- **Automatic backend selection** (always uses fastest available)
- **NumPy vectorization** for theta/delta symbols
- **Multi-variable summations** (arbitrary N nested summations)

---

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

Or install everything at once:
```bash
pip install -r requirements.txt
```

---

## Quick Start

Parallel evaluation is the default — no configuration needed.  Use the public
API in `src/api.py`:

```python
from src.api import load_network

snet = load_network("drawn_graph.graphml")
formula = snet.evaluate_symbolic()

# Default: auto-selects JAX GPU → JAX CPU → multiprocessing → serial
result = formula.evaluate_numeric()
```

---

## Backend Selection

The evaluator picks the fastest available backend automatically:

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | JAX GPU | JAX installed + GPU detected |
| 2 | JAX CPU | JAX installed, no GPU |
| 3 | Multiprocessing | JAX not installed |
| 4 | Serial | Fallback / debug |

### Choosing a Backend via the API

Pass `backend` and `max_two_j` directly to `evaluate_numeric()` or `evaluate_batch()`:

```python
# GPU (requires JAX / jax-metal)
result = formula.evaluate_numeric(backend="jax")

# Parallel CPU
result = formula.evaluate_numeric(backend="multiprocessing")

# Single-threaded — easiest to debug
result = formula.evaluate_numeric(backend="serial")

# Large spins (default max_two_j=200 → j up to 100; raise as needed)
result = formula.evaluate_numeric(backend="jax", max_two_j=2000)
```

The same keywords work for batch evaluation:

```python
results = formula.evaluate_batch(
    [[SpinArg("j_1", v)] for v in [0.5, 1.0, 1.5, 2.0]],
    backend="multiprocessing",
)
```

`evaluate_batch()` reuses a single evaluator instance across all entries —
much faster than calling `evaluate_numeric()` in a loop.

---

## Performance

### Expected Speedups (vs. serial)

| Backend | Typical Speedup | Hardware Required |
|---------|----------------|-------------------|
| JAX GPU | 10–100x | NVIDIA GPU or Apple M1/M2/M3 |
| JAX CPU | 3–10x | Any CPU |
| Multiprocessing | 2–8x | Multi-core CPU |

### Benchmarking

Compare all available backends on your hardware:

```python
from src.spin_evaluator import benchmark_backends

times = benchmark_backends(canonical_terms, max_two_j=200)

# Output:
# ======================================================================
# BENCHMARK SUMMARY
# ======================================================================
#   jax: 0.845s (speedup: 6.19x)
#   multiprocessing: 1.234s (speedup: 4.24x)
#   serial: 5.234s (speedup: 1.00x)
#
# Best backend: jax (0.845s)
# ======================================================================
```

---

## Checking Available Backends

```python
from src.spin_evaluator import JAX_AVAILABLE, JAX_GPU_AVAILABLE

print(f"JAX installed:     {JAX_AVAILABLE}")
print(f"JAX GPU available: {JAX_GPU_AVAILABLE}")
```

---

## Implementation Details

### Parallel Strategy

**Single summation variable:**
- Range split into N chunks (N = number of workers)
- Each worker evaluates its chunk independently
- Results summed at the end

**Multiple summation variables:**
- `itertools.product()` generates all combinations
- Full combination space chunked across workers
- Embarrassingly parallel — no inter-process communication

### Limitations

**JAX backend:**
- Wigner 6j symbols still use `pywigxjpf` (C++ library, already optimized)
- JAX accelerates theta/delta computations
- Full GPU-native 6j would require a JAX-native implementation

**Memory:**
- For 5+ summation variables with wide ranges, the combination list may be large
- Use `backend="serial"` or reduce `max_two_j` in memory-constrained environments

---

## Troubleshooting

### "JAX backend requested but JAX not installed"

```bash
pip install jax        # CPU only
pip install jax-metal  # Apple Silicon GPU
```

### Slow performance on Apple Silicon

Verify `jax-metal` is installed (not the CPU-only `jax`):
```bash
pip uninstall jax
pip install jax-metal
```

### Out of memory errors

Lower `max_two_j` or switch to serial:
```python
result = formula.evaluate_numeric(backend="multiprocessing", max_two_j=100)
result = formula.evaluate_numeric(backend="serial", max_two_j=100)
```

### Multiprocessing freeze in Jupyter

If the notebook hangs when using `backend="multiprocessing"`, add this once
before the first `evaluate_numeric()` call:
```python
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
```

---

## Advanced / Low-Level API

The public API (`Formula.evaluate_numeric`) covers most use cases.  For
fine-grained control — custom worker counts, manual table management — use
`SpinNetworkEvaluator` directly:

```python
from src.spin_evaluator import SpinNetworkEvaluator
from multiprocessing import cpu_count

evaluator = SpinNetworkEvaluator(
    max_two_j=2000,
    backend="multiprocessing",
    n_workers=cpu_count() - 1,   # leave one core free
)
# ... use evaluator ...
evaluator.cleanup()              # always call cleanup to free wigxjpf tables
```

### `SpinNetworkEvaluator` reference

```python
SpinNetworkEvaluator(max_two_j=200, backend='auto', n_workers=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_two_j` | int | 200 | Maximum 2×j value; tables scale as O(j²) |
| `backend` | str | `'auto'` | `'auto'`, `'jax'`, `'multiprocessing'`, `'serial'` |
| `n_workers` | int | None | CPU workers (defaults to `cpu_count() - 1`) |

### `benchmark_backends` reference

```python
from src.spin_evaluator import benchmark_backends
times = benchmark_backends(canonical_terms, max_two_j=200)
# Returns: dict mapping backend name → execution time (seconds)
```

---

## Contributing

To add a new backend:

1. Add imports at the top of `src/spin_evaluator.py`
2. Update backend selection logic in `SpinNetworkEvaluator.__init__`
3. Implement `_evaluate_sum_BACKEND()` method
4. Update this document and add tests

See `_evaluate_sum_jax()` and `_evaluate_sum_parallel()` for examples.
