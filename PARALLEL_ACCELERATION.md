# Evaluation Backends and Performance

How the Wigner 6j summations are evaluated, what the options actually cost, and
why the default is what it is. **Every number here is measured** — reproduce it
with `python scripts/benchmark_backends.py`.

## Summary

- **`backend="auto"` means serial.** For almost every network that is the
  fastest choice.
- **There is no GPU support**, and none is possible without replacing wigxjpf.
  JAX support was removed: it cannot trace a C library.
- **CPU parallelism helps only above ~350,000 summation terms** (measured
  break-even). Below that it is slower — by up to three orders of magnitude.
- Serial and parallel results are **bitwise identical**.

This document previously advertised 10–100× GPU speedups. Those numbers were
never measured and the GPU code path did not exist: `_evaluate_sum_jax` was a
stub that called the serial routine. JAX has since been removed entirely.

---

## Installation

Nothing extra is needed:

```bash
pip install -r requirements.txt
```

There is no optional GPU extra: **no array framework can accelerate this**,
because the 6j symbols come from a C library. See below.

---

## Quick Start

No configuration needed. Use the public API in `src/api.py`:

```python
from src.api import load_network

snet = load_network("drawn_graph.graphml")
formula = snet.evaluate_symbolic()

# Default: serial
result = formula.evaluate_numeric()
```

---

## Backend Selection

`backend="auto"` resolves as follows:

| Priority | Backend | Condition |
|----------|---------|-----------|
| 1 | Serial | Always — this is what `"auto"` selects |
| 2 | Multiprocessing | Only when requested explicitly *and* the work is large enough to beat the ~1 s startup cost |

### Choosing a Backend via the API

Pass `backend` and `max_two_j` directly to `evaluate_numeric()` or `evaluate_batch()`:

```python
# Default: serial
result = formula.evaluate_numeric()

# Parallel CPU — only worth it for very large summations, and only from a
# script whose entry point is guarded by  if __name__ == "__main__":
result = formula.evaluate_numeric(backend="multiprocessing")

# Explicitly single-threaded
result = formula.evaluate_numeric(backend="serial")

# Large spins (default max_two_j=200 → j up to 100; raise as needed)
result = formula.evaluate_numeric(max_two_j=2000)
```

The same keywords work for batch evaluation:

```python
results = formula.evaluate_batch(
    [[UnitArg("j_1", v)] for v in [0.5, 1.0, 1.5, 2.0]],
    backend="multiprocessing",
)
```

`evaluate_batch()` reuses a single evaluator instance across all entries —
much faster than calling `evaluate_numeric()` in a loop.

---

## Performance

### There is no GPU speedup, and CPU parallelism rarely helps

Measured, not estimated. Reproduce with `python scripts/benchmark_backends.py`.

**No array framework can accelerate this calculation.** The dominant cost is
the Wigner 6j symbol, which comes from wigxjpf — a C library that cannot be
traced, `jit`-ed or `vmap`-ed. There is no GPU code path for it and none is
possible without reimplementing 6j evaluation from scratch.

**CPU parallelism has a large fixed cost.** Each worker must be spawned and
must re-allocate its own wigxjpf tables:

| Quantity | Measured (8-core arm64 macOS, `max_two_j=200`) |
|----------|-----------------------------------------------|
| Pool startup + wigxjpf init, 7 workers | ~0.8–1.0 s |
| Serial cost per summation term | ~2.6 µs |
| **Break-even** | **~350,000 summation terms** |
| Speedup at 2.7M terms | 2.55x |

So parallelism only pays off for the very large multi-variable summations
where the number of terms is the product of several F-variable ranges. Below
the break-even it is *slower* — by up to three orders of magnitude on an
ordinary formula.

Because of this, **`backend='auto'` resolves to `'serial'`.** Even when you
ask for `'multiprocessing'` explicitly, the evaluator times a pilot slice of
the outer summation first and stays serial unless the extrapolated work
clearly exceeds the startup cost.

### Using the multiprocessing backend

```python
result = formula.evaluate_numeric(backend="multiprocessing")
```

Two constraints, both enforced automatically:

1. **Your script must guard its entry point.** On macOS and Windows the
   `spawn` start method re-imports `__main__` in every worker, so an
   unguarded script re-runs itself once per worker:

   ```python
   if __name__ == "__main__":
       main()
   ```

2. **It is disabled in notebooks and interactive sessions**, where that
   re-import cannot work at all (the classic "multiprocessing hangs in
   Jupyter" failure). The evaluator detects this and falls back to serial
   rather than hanging.

### Correctness

Parallel and serial results are **bitwise identical**, verified in
`tests/test_validation.py::TestBackendDispatch`. Splitting is only applied
when the outermost summation enters the expression linearly (checked with the
`ast` module); a sum that is squared, or buried in a `safe_div(...)` argument
as transition-probability formulas are, is evaluated serially rather than
chunked incorrectly.

```bash
python scripts/check_backends.py      # verifies serial == parallel
python scripts/benchmark_backends.py  # reproduces the table above
```

---

## Checking What Will Actually Run

```python
from src.spin_evaluator import SpinNetworkEvaluator, _multiprocessing_is_usable

ev = SpinNetworkEvaluator(max_two_j=2, backend="auto", verbose=False)
print(f"'auto' resolves to         : {ev.backend!r}")
ev.cleanup()

# False in notebooks and interactive sessions -- see Troubleshooting.
print(f"multiprocessing usable here: {_multiprocessing_is_usable()}")
```

---

## Implementation Details

### Parallel Strategy

The **outermost** summation is split into contiguous chunks, one per worker;
nested summations run in full inside each chunk. Partial results are added at
the end.

This is exact because chunking is only applied when the expression is a
product in which that sum appears exactly once, so

    value = C x S,   S = sum over the range,   C independent of it

and therefore `sum over chunks of (C x S_chunk) = C x S`. The check is done on
the parsed syntax tree (`FormulaEvaluator._outer_sum_is_a_linear_factor`). A
sum that is squared, divided by, or passed as a function argument — which is
what `calculate_probability()` produces via `safe_div(...)` — is **not**
chunked; the evaluator falls back to serial rather than risk a wrong number.

Workers are module-level functions initialised through a `Pool` initializer, so
each process allocates its wigxjpf tables exactly once. (An earlier version
passed a nested closure to `Pool.map`; closures cannot be pickled, so that path
could never run at all.)

### Limitations

**No GPU path:** Wigner 6j symbols come from `pywigxjpf`, a C library. Array
frameworks cannot trace into it, so the dominant cost cannot be moved to a GPU
without reimplementing 6j evaluation from scratch.

**Startup cost:** ~1 s, dominated by spawning processes and re-allocating
wigxjpf tables per worker. This is why the break-even is so high.

**Memory:** for 5+ summation variables with wide ranges the combination space
is large. Reduce `max_two_j`, or narrow the F-variable ranges.

---

## Troubleshooting

### "It is not using my GPU"

Correct, and it never will. See the Performance section: the 6j symbols come
from a C library that array frameworks cannot trace into. JAX support was
removed for exactly this reason.

### "multiprocessing is not making it faster"

Expected below ~350,000 summation terms. The evaluator times a pilot slice and
deliberately stays serial when parallelism would lose. Run
`python scripts/benchmark_backends.py` to see the break-even on your machine.

### "multiprocessing does nothing in my notebook"

Also expected, and deliberate. Under the `spawn` start method every worker
re-imports `__main__`; a Jupyter kernel has no importable `__main__`, so the
workers fail and the pool retries forever. The library detects this and falls
back to serial instead of hanging.

To use multiple cores, run from a script file with a guarded entry point:

```python
if __name__ == "__main__":
    main()
```

Without that guard, each worker re-imports and **re-runs your whole script**.

### Out of memory errors

Lower `max_two_j`:
```python
result = formula.evaluate_numeric(max_two_j=100)
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
SpinNetworkEvaluator(max_two_j=200, backend='auto', n_workers=None, verbose=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_two_j` | int | 200 | Maximum 2×j value; tables scale as O(j²) |
| `backend` | str | `'auto'` | `'auto'` (= serial), `'serial'`, `'multiprocessing'` |
| `n_workers` | int | None | CPU workers (defaults to `cpu_count() - 1`) |
| `verbose` | bool | True | Print backend/table messages. Workers pass False so a parallel run does not emit one banner per core |

### Benchmark and verification scripts

```bash
python scripts/check_backends.py      # asserts serial == parallel, bitwise
python scripts/benchmark_backends.py  # measures the break-even on your machine
```

Both must be run as scripts, not from a notebook or a heredoc — they use the
multiprocessing backend, which needs an importable `__main__`.

---

## Contributing

To add a new backend:

1. Add imports at the top of `src/spin_evaluator.py`
2. Update backend selection logic in `SpinNetworkEvaluator.__init__`
3. Implement `_evaluate_sum_BACKEND()` method
4. Update this document and add tests

See `_evaluate_sum_parallel()` and `FormulaEvaluator._try_parallel_evaluate()` for examples.
