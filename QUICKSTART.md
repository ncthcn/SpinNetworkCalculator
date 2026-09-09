# Quick Start Guide

**For collaborators who just want to compute spin network norms without understanding all the details.**

## Installation (One Time Only)

```bash
# 1. Navigate to the project folder
cd SpinNetworkCalculator

# 2. Create virtual environment
python3 -m venv myenv

# 3. Activate it
source myenv/bin/activate  # On Mac/Linux
# OR
myenv\Scripts\activate     # On Windows

# 4. Install dependencies
pip install -r requirements.txt

```

**Note on speed:** there is no GPU acceleration. The dominant cost is the
Wigner 6j symbol, computed by wigxjpf — a C library that array frameworks
cannot trace into. CPU parallelism
(`backend="multiprocessing"`) only pays off above roughly 350,000 summation
terms; below that it is slower, so the default is serial. Measure it yourself
with `python scripts/benchmark_backends.py`.

## Usage (Every Time)

### Option A: Draw a new spin network and compute its norm

```python
from src.api import new_network

snet = new_network()                     # opens drawing GUI (press S when done)
formula = snet.evaluate_symbolic()       # symbolic reduction
formula.save("result.pdf", "pdf")        # save PDF
result = formula.evaluate_numeric()      # numerical value
print(result)
```

### Option B: Load an existing GraphML file

```python
from src.api import load_network

snet = load_network("drawn_graph.graphml")
formula = snet.evaluate_symbolic()
result = formula.evaluate_numeric()
print(result)
```

### Option C: Reload a previously computed expression

```python
from src.api import Formula

formula = Formula.load("canon_norm_expression.txt")
result = formula.evaluate_numeric()
print(result)
```

### Option D: Transition probability between two networks

```python
from src.api import load_network, calculate_probability

n1 = load_network("drawn_graph.graphml")
n2 = n1.transition_to()          # opens GUI — modify graph, press S to save

formula = calculate_probability(n1, n2)   # symbolic, like evaluate_symbolic()
p = formula.evaluate_numeric()
print(f"Transition probability: {p}")
```

In the GUI:
- Orange nodes/edges are open ends
- Press **C** with two open nodes selected to reconnect them
- Press **E** to add a new edge
- Press **S** to save and exit

## What You Get

### Files Generated

**From `new_network()` / `load_network()` + `.save()`:**
- `drawn_graph.graphml` — the spin network graph

**From `formula.save()`:**
- `<name>.pdf` — LaTeX-rendered canonical expression
- `<name>.txt` — plain-text expression (reload via `Formula.load()`)

**From `n1.transition_to()`:**
- `transition_to_graph.graphml` — the modified (child) graph
- `transition_to_graph_transition.json` — structural metadata (added edges, reconnections)

### Console Output During Evaluation:
```
Using serial backend (pass backend='multiprocessing' for very large summations)
Initializing wigxjpf tables for max 2j = 200...
✓ Wigxjpf initialized and ready

SPIN NETWORK NORM = 6.658558117818342e+01
```

The norm is returned as a non-negative number: the individual factors (θ, Δ,
the (−1)^… prefactors) keep their signs throughout the calculation, and the
modulus is applied once at the very end.

## Common Issues

### "File not found"
→ Run `new_network()` first to create and save a network

### "Triangular condition not satisfied"
→ Your edge spins don't satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂ at some node

### Result is zero
→ Your spin network configuration is forbidden by the rules of SU(2)

### Non-planar graph warning
→ Saves `{input_basename}_kuratowski.png` showing the K₅ or K₃,₃ subdivision, then continues with cycle-basis fallback. Use `snet.evaluate_symbolic(strict_planarity=True)` to abort instead.

### "RuntimeError: Transition not completed"
→ The transition GUI was closed without pressing **S (Save)**. Run `n1.transition_to()` again.

### "Memory error" with large spins
→ Pass `max_two_j` directly:
```python
result = formula.evaluate_numeric(max_two_j=2000)  # j up to 1000
```

## Advanced: Large Spin Values

The calculator handles large spins (j up to 1000+) automatically:
- **Theta symbols**: log-gamma for j > 100 (numerically stable)
- **Delta symbols**: Δⱼ = (-1)^(2j) × (2j+1)

No special configuration needed — just specify `max_two_j` for memory allocation.

## Need More Help?

- Full guide: [README.md](README.md)
- Backends and measured performance: [PARALLEL_ACCELERATION.md](PARALLEL_ACCELERATION.md)
- Validation against known results: `pytest tests/test_validation.py -v`
- Graph comparison: [scripts/README_COMPARISON.md](scripts/README_COMPARISON.md)
- Interactive tutorial: `tutorial.ipynb`
