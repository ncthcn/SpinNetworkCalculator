# Spin Networks Calculator

[![Tests](https://github.com/ncthcn/SpinNetworkCalculator/actions/workflows/tests.yml/badge.svg)](https://github.com/ncthcn/SpinNetworkCalculator/actions/workflows/tests.yml)

A computational tool for calculating spin network norms and probabilities. This project performs symbolic graph reduction and numerical evaluation of spin networks using a combinatoric algorithm inspired by the Decomposition Theorem [1,2].

---

## Documentation Quick Links

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes (for collaborators)
- **[PARALLEL_ACCELERATION.md](PARALLEL_ACCELERATION.md)** - Backends and measured performance
- **[scripts/README_COMPARISON.md](scripts/README_COMPARISON.md)** - Graph comparison workflow
- **This README** - Comprehensive documentation
- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ncthcn/SpinNetworkCalculator/Cleaning?urlpath=%2Fdoc%2Ftree%2Ftutorial.ipynb)

---

## Table of Contents

- [What is this?](#what-is-this)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python Library API](#python-library-api)
- [Usage Guide](#usage-guide)
- [Understanding the Output](#understanding-the-output)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## What is this?

**Spin networks** are combinatorial structures first presented by Roger Penrose. This tool:

1. **Takes a graph as input** where:
   - Nodes are trivalent
   - Edges have half-integer labels (spin values)

2. **Performs symbolic computation**:
   - Applies F-moves and triangle reductions
   - Generates canonical expressions with Wigner 6j symbols
   - Returns a `Formula` object that can be saved as PDF or plain text

3. **Computes numerical values**:
   - Uses high-performance C++ backend (wigxjpf)
   - Handles large spin values efficiently
   - Returns the norm as a Python `float` via `formula.evaluate_numeric()`


---

## Installation

### Prerequisites

- **Python 3.7+**
- **pip** (Python package manager)

### Step 1: Install Python Dependencies

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

This installs:
- Core dependencies: networkx, matplotlib, sympy, pybind11, pywigxjpf
- NumPy for vectorization
- SciPy for `gammaln`, used by the theta symbol at large spins
- sympy, also used by the validation tests as an independent 6j reference

**There is no GPU acceleration.** The dominant cost is the Wigner 6j symbol,
computed by wigxjpf — a C library that array frameworks such as JAX cannot
trace, `jit` or `vmap`. JAX support was removed because it could not help; see
[Computation Backends](#computation-backends).

**Alternative (manual installation):**
```bash
pip install networkx matplotlib sympy pybind11 pywigxjpf numpy scipy
```

### Step 2: Verify Installation

```bash
python -c "import pywigxjpf; print('✓ wigxjpf installed successfully')"
```

If you see the success message, you're ready to go.

---

## Quick Start

### 1. Create a Spin Network

```python
from src.api import new_network, load_network

# Option A: draw interactively (opens a GUI window)
# Note: in Jupyter, run  %gui tk  in a cell first
snet = new_network()

# Option B: load from an existing .graphml file
snet = load_network("drawn_graph.graphml")
```

**GUI controls (when using `new_network()`):**

| Key | Action |
|-----|--------|
| N | Add Node |
| E | Add Edge → click two nodes, enter spin value |
| M | Move Node |
| D | Delete Node |
| X | Delete Edge |
| Z | Undo |
| S | Save & Exit |

### 2. Compute the Symbolic Norm

```python
formula = snet.evaluate_symbolic()
print(formula)
# Formula(from graph, terms=1, free=[], assigned=[])

# If the graph has symbolic edge labels, assign numeric values first
args = snet.get_args()           # list of free spin variables
args[0].value = 1.5              # assign a value directly
snet.set_args(args)              # applies and validates triangular inequality

formula = snet.evaluate_symbolic()   # re-runs with updated labels
```

### 3. Evaluate Numerically

```python
result = formula.evaluate_numeric()
print(result)   # e.g. -60.0

# Save the expression for reference
formula.save("norm.pdf", "pdf")   # LaTeX-rendered PDF
formula.save("norm.txt", "txt")   # plain-text Python expression
```

For the full API reference see the [Python Library API](#python-library-api) section below.

---

## Python Library API

In addition to the CLI scripts, the project provides a clean Python library API in `src/api.py` designed for use in **Jupyter Notebooks** or downstream Python code.  The full mathematical pipeline runs in memory — no intermediate files are created.

### Core Classes

| Class | Purpose |
|---|---|
| `UnitArg` | One free spin variable (edge label + assigned value) |
| `Graph` | Trivalent graph with evaluation methods and formula cache |
| `SpinNetwork` | User-facing wrapper around `Graph` (extend with future methods here) |
| `Formula` | Symbolic norm expression; evaluates numerically and saves to file |

### Full Workflow

```python
from src.api import new_network, load_network, UnitArg

# ── Option A: draw a new graph interactively ───────────────────────────────
# Opens the graph editor GUI. Close it with S (Save & Exit).
# Note: in Jupyter, run  %gui tk  first.
snet = new_network()

# ── Option B: load an existing .graphml file ───────────────────────────────
snet = load_network("drawn_graph.graphml")

# ── Inspect and edit ───────────────────────────────────────────────────────
snet.display()           # read-only visual inspector
snet.modify()            # interactive editor (invalidates formula cache)

# ── Manage free (symbolic) spin variables ──────────────────────────────────
args = snet.get_args()   # list of UnitArg objects, one per symbolic edge label
for a in args:
    print(a.label, a.value, a.is_numeric)

args[0].value = 1.5      # assign a concrete spin value directly
snet.set_args(args)      # validates triangular inequality, then applies changes

# ── Symbolic evaluation (expensive; result is cached) ──────────────────────
formula = snet.evaluate_symbolic()
print(formula)           # Formula(from graph, terms=3, free=[], assigned=[])

# ── Save the symbolic expression ───────────────────────────────────────────
formula.save("result.txt", "txt")   # plain-text Python expression
formula.save("result.pdf", "pdf")   # LaTeX-rendered PDF

# ── Reload from file (supports evaluate_numeric, not PDF saving) ───────────
from src.api import Formula
f2 = Formula.load("result.txt")

# ── Numerical evaluation ───────────────────────────────────────────────────
result = formula.evaluate_numeric()                          # default backend (serial)
result = formula.evaluate_numeric([UnitArg("j_1", 2.0)])    # pass overrides directly
result = formula.evaluate_numeric(backend="multiprocessing") # only for very large sums
result = formula.evaluate_numeric(backend="serial")          # single-threaded
result = formula.evaluate_numeric(max_two_j=2000)            # allow large spins (> j=100)

# ── Batch evaluation over a range of spins ─────────────────────────────────
# evaluate_batch shares one evaluator instance — much faster than looping evaluate_numeric()
results = formula.evaluate_batch([
    [UnitArg("j_1", 0.5)],
    [UnitArg("j_1", 1.0)],
    [UnitArg("j_1", 1.5)],
    [UnitArg("j_1", 2.0)],
])
# results == [f(0.5), f(1.0), f(1.5), f(2.0)]

results = formula.evaluate_batch(args_list, backend="multiprocessing")  # parallel batch

# ── Persist the graph ──────────────────────────────────────────────────────
snet.save("my_network.graphml")
```

### Caching and Idempotency

`evaluate_symbolic()` is expensive (F-moves, triangle reductions, canonicalisation).  Its result is cached inside `Graph`.  The cache is **automatically invalidated** when:
- `snet.modify()` is called (graph structure changed)
- `snet.set_args(args)` is called (edge labels changed)

Re-running a Jupyter cell that calls `evaluate_symbolic()` on an unchanged graph returns the cached `Formula` instantly.

### UnitArg: Python vs C++ idioms

`UnitArg` uses Python's `@dataclass` and `@property` instead of C++ getters/setters:

```python
# C++ style (NOT how this works)
arg.getValue()     # ✗
arg.setValue(1.5)  # ✗

# Python style (correct)
arg.value          # read  → @property is_numeric tells you if it's a number
arg.value = 1.5    # write → direct attribute assignment
arg.is_numeric     # True after assigning a float
```

### Computation Backends

Both `evaluate_numeric()` and `evaluate_batch()` accept a `backend` keyword. **The
default is fine for essentially all networks** — the figures below are measured, and
reproducible with `python scripts/benchmark_backends.py`.

| Backend | What it does |
|---|---|
| `"auto"` (default) | Resolves to `"serial"` |
| `"serial"` | Single thread |
| `"multiprocessing"` | Splits the outermost summation across cores. Times a pilot slice first and stays serial unless the work clearly exceeds the ~1 s startup cost |

**There is no GPU acceleration, and none is possible without replacing wigxjpf.**
The dominant cost is the Wigner 6j symbol, computed by wigxjpf — a C library that
array frameworks cannot trace into.

**CPU parallelism only pays off for very large summations.** Each worker must be
spawned and must re-allocate its own wigxjpf tables:

| Quantity | Measured (8-core arm64 macOS, `max_two_j=200`) |
|---|---|
| Pool startup + wigxjpf init, 7 workers | ~0.8–1.0 s |
| Serial cost per summation term | ~2.6 µs |
| **Break-even** | **~350,000 summation terms** |
| Speedup at 2.7 M terms | 2.55× |

Below that break-even, parallelism is *slower* — by up to three orders of magnitude.

Using `"multiprocessing"` has two constraints, both enforced automatically:

1. Your script must guard its entry point with `if __name__ == "__main__":`, because
   the `spawn` start method re-imports `__main__` in every worker.
2. It is disabled in notebooks and interactive sessions, where that re-import cannot
   work at all (the classic "multiprocessing hangs in Jupyter"), and falls back to
   serial rather than hanging.

Serial and parallel results are **bitwise identical** — verified in
`tests/test_validation.py::TestBackendDispatch`.

The `max_two_j` parameter pre-allocates wigxjpf tables for spins up to `max_two_j/2`
(default `200` → j up to 100). Raise it for larger spins; lower it to save memory:

```python
# Large spins
result = formula.evaluate_numeric(max_two_j=2000)

# Memory-constrained environment
result = formula.evaluate_numeric(max_two_j=100)

# Very large multi-variable summation, from a guarded script
result = formula.evaluate_numeric(backend="multiprocessing")
```

See **[PARALLEL_ACCELERATION.md](PARALLEL_ACCELERATION.md)** for the full measurements.

---

## Usage Guide

### Reconnection Probability Workflow

The transition probability between two spin network states is computed via the `calculate_probability()` function.  The transition itself is created interactively with `transition_to()`, which opens a GUI to modify the graph (reconnect open ends, add edges) and saves the resulting child network.

`calculate_probability(n1, n2)` mirrors `Graph.evaluate_symbolic()`: it returns a symbolic `Formula`, not a number.  Call `formula.evaluate_numeric()` for a single value or `formula.evaluate_batch(args_list)` to scan many spin assignments efficiently (one shared evaluator for the whole batch) — exactly like any other `Formula`.

```python
from src.api import load_network, calculate_probability, UnitArg, TreeVisualizer

n1 = load_network("drawn_graph.graphml")

# Open GUI: select open nodes, press C to reconnect, S to save
n2 = n1.transition_to()

# Build the symbolic probability formula, then evaluate it
formula = calculate_probability(n1, n2)
p = formula.evaluate_numeric()
print(f"P = {p}")

# Scan a batch of spin assignments (e.g. for a symbolic edge "j_1")
args_list = [[UnitArg("j_1", v)] for v in (0.5, 1.0, 1.5, 2.0)]
probs = formula.evaluate_batch(args_list)

# Visualise the genealogy tree
TreeVisualizer.display_tree(n1)
```

The probability formula is:
```
P = |Δ(c₁)⋯Δ(cₙ) / [Θ(c₁,s₁,t₁)⋯Θ(cₙ,sₙ,tₙ)]  ×  ||G₂|| / (||G₁|| × ||GΔ||)|
```

where `cᵢ` are new edge labels from reconnections, `sᵢ,tᵢ` are the reconnected edges, and `GΔ` is the subgraph of explicitly added edges.  A zero denominator (`||G₁||×||GΔ||`) or an inadmissible `Θ(cᵢ,sᵢ,tᵢ)` contributes 0 rather than raising a division error.

**In the GUI:**
- Orange nodes/edges are open ends
- Select two open nodes → press **C** to reconnect, enter the new edge label
- Press **E** to add a new edge
- Press **S** to save and exit

### Creating and Editing a Spin Network

```python
from src.api import new_network, load_network

# Draw a new graph interactively
snet = new_network()          # blocks until you press S (Save & Exit)

# Load an existing graph
snet = load_network("my_network.graphml")

# Inspect (read-only viewer) or edit (interactive editor)
snet.display()
snet.modify()                 # invalidates the formula cache on close

# Save to disk at any time
snet.save("my_network.graphml")
```

**Graph constraints enforced by the editor:**
- Trivalent nodes — at most 3 edges per node
- Half-integer spin labels: 0, 0.5, 1, 1.5, …
- Triangular inequality at each complete vertex: |j₁−j₂| ≤ j₃ ≤ j₁+j₂
- Symbolic labels (e.g. `j_1`, `a`) are accepted for parametric networks

### Large Spin Values

The evaluator handles large spins automatically using log-space arithmetic:
- **Theta symbols**: `scipy.special.gammaln` for factorials in log-space
- **Delta symbols**: `exp(2j × log(2j+1))` to avoid overflow
- **Memory**: wigxjpf tables scale as O(j²); j=1000 requires ~hundreds of MB

For very large spin values the default `max_two_j=200` (j up to 100) may need raising.
Pass it directly through the API:

```python
result = formula.evaluate_numeric(max_two_j=2000)            # j up to 1000
results = formula.evaluate_batch(args_list, max_two_j=2000)
```

If you need even finer control, access `FormulaEvaluator` directly:

```python
from src.spin_evaluator import FormulaEvaluator
evaluator = FormulaEvaluator(max_two_j=4000)
result = evaluator.evaluate(formula_string, variables={"j_1": 500.0})
evaluator.cleanup()
```

---

## Understanding the Output

### Formula object

`snet.evaluate_symbolic()` returns a `Formula` object.  Printing it shows a
summary of the canonical expression:

```python
formula = snet.evaluate_symbolic()
print(formula)
# Formula(from graph, terms=3, free=["j_1"], assigned=[])
```

`terms` is the number of canonical terms in the sum.
`free` lists spin variables that still need a numeric value.
`assigned` lists variables that have already been set.

### Saved files

```python
formula.save("norm.pdf", "pdf")   # LaTeX-rendered PDF
formula.save("norm.txt", "txt")   # plain-text Python expression
```

**PDF** shows the canonical form with Wigner 6j symbols, theta/delta symbols,
sign factors, and proper mathematical notation.

**TXT** contains a plain-text Python expression that can be reloaded:

```python
from src.api import Formula
f2 = Formula.load("norm.txt")
result = f2.evaluate_numeric(args)
```

### Numerical result

`formula.evaluate_numeric()` returns a Python `float`:

```python
result = formula.evaluate_numeric()
print(result)   # e.g. -60.0
```

**Interpreting the value:**
- **Non-zero**: The spin network is physically allowed
- **Very small (~10⁻¹⁰)**: May indicate numerical precision issues
- **Zero**: The configuration violates SU(2) coupling rules

---

## Technical Details

### Mathematical Background

A **spin network** is a graph with:
- Edges labeled by spins (half-integers: 0, 1/2, 1, 3/2, 2, ...)
- Nodes satisfying the **triangle inequality**: For edges j₁, j₂, j₃ meeting at a node:
  ```
  |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
  ```

The **norm** is computed by taking a copy of the network, gluing it along its opend ends, expanding each j-edge as an antisymmetrised set of 2j strands, counting the number of strand loops formed and assigning a (-2) value to each. The norm will be found as a product of:
- **Wigner 6j symbols**: SU(2) recoupling coefficients
- **Theta symbols**: θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
- **Delta symbols**: Δⱼ = (-1)^(2j) × (2j+1)
- **Sign factors**: (-1)^(...)

### Algorithm Overview

```
Input Graph
    ↓
Glue Open Edges (create closed graph)
    ↓
Apply F-moves (reduces (n>3)-cycles)
    ↓
Triangle Reductions (reduces 3-cycles)
    ↓
Expand 6j → W6j (with theta/delta factors)
    ↓
Canonicalize (combine duplicates, 24-fold tetrahedral 6j symmetry)
    ↓
Numerical Evaluation (compute 6j values via wigxjpf)
    ↓
Final Result
```

---

## Troubleshooting

### Problem: "File not found"

**Error:**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'drawn_graph.graphml'
```

**Solution:** Create a graph first with `new_network()`, or check the path passed to `load_network()`.

---

### Problem: "wigxjpf not installed"

**Error:**
```
ModuleNotFoundError: No module named 'pywigxjpf'
```

**Solution:**
```bash
pip install pywigxjpf
```

If that fails, you may need to install from source:
```bash
# Download wigxjpf
curl -L https://fy.chalmers.se/subatom/wigxjpf/wigxjpf-1.11.tar.gz -o wigxjpf.tar.gz
tar -xzf wigxjpf.tar.gz
cd wigxjpf-1.11

# Build and install
make
python setup.py install
```

---

### Problem: "Memory error" or "Too large"

**Error:**
```
MemoryError: Cannot allocate wigxjpf tables
```

**Cause:** The maximum spin value is too large (tables scale as O(j²))

**Solution:** Use `FormulaEvaluator` directly with a lower `max_two_j`:
```python
from src.spin_evaluator import FormulaEvaluator
evaluator = FormulaEvaluator(max_two_j=200)   # default; lower to save memory
result = evaluator.evaluate(formula_string)
evaluator.cleanup()
```

---

## File Structure

```
Spin_Networks_Project_full/
│
├── Documentation
│   ├── README.md                        # This file - comprehensive guide
│   ├── QUICKSTART.md                    # 5-minute quick start
│   ├── PARALLEL_ACCELERATION.md         # Backends and measured performance
│   └── requirements.txt                 # Python dependencies
│
├── GUI modules (scripts/)               # used internally by src/api.py
│   ├── graph.py                         # GraphEditor class  (new_network())
│   ├── inspect_graph.py                 # GraphInspector class  (Graph.display())
│   ├── modify_graph.py                  # GraphModifier class  (Graph.modify())
│   ├── transition_to.py                 # Transition GUI  (SpinNetwork.transition_to())
│   │
│   └── Standalone CLI tools
│       ├── compute_probability.py           # Single reconnection probability (CLI)
│       ├── compute_all_probabilities.py     # Full probability distribution (CLI)
│       ├── compute_symbolic_probability.py  # Symbolic probability formula (CLI)
│       ├── check_backends.py                # Verifies serial == parallel
│       ├── benchmark_backends.py            # Measures the parallel break-even point
│       ├── compare_graphs.py                # Automated graph comparison workflow
│       ├── compare_graphs_cli.py            # Graph comparison (CLI)
│       └── README_COMPARISON.md             # Graph comparison workflow docs
│
├── Core Library (src/)
│   ├── api.py                   # Public library API (SpinNetwork, Graph, Formula, UnitArg)
│   ├── evolution.py             # Transition class, LineageError
│   ├── probability.py           # calculate_probability()
│   ├── visualizer.py            # TreeVisualizer
│   ├── drawing.py               # Graph visualization, Kuratowski plots
│   ├── gluer.py                 # Graph gluing operations
│   ├── graph_reducer.py         # F-moves and triangle reductions
│   ├── norm_reducer.py          # Canonicalization (tetrahedral 6j symmetry)
│   ├── spin_evaluator.py        # Numerical evaluation with wigxjpf
│   ├── LaTeX_rendering.py       # PDF generation
│   ├── utils.py                 # Utility functions
│   ├── orientation.py           # Reference orientation calculations
│   └── reduction_animator.py    # Reduction step animation (GIFs)
│
├── Generated Files (names are user-controlled via the API)
│   ├── drawn_graph.graphml                      # saved by new_network() / Graph.save()
│   ├── <name>.pdf                               # saved by formula.save("name.pdf","pdf")
│   ├── <name>.txt                               # saved by formula.save("name.txt","txt")
│   │
│   └── Transition outputs  (from transition_to())
│       ├── transition_to_graph.graphml          # child graph
│       ├── transition_to_graph_transition.json  # structural metadata
│       ├── graph_snapshots/graph.png
│       └── {input_basename}_kuratowski.png      # K₅/K₃,₃ subgraph (non-planar only)
│
└── Other
    ├── tests/                   # Test suite  (run: pytest tests/)
    │   └── test_validation.py   # Validation vs sympy / closed forms / known norms
    └── .gitignore               # Git ignore rules
```

---

## For Researchers

### Citation

The paper using this code and numerical results produced by it is in the writing process.
This section will be updated as soon as the paper is submitted.

---

### Validation

The numerical core is checked against sources independent of this codebase:

```bash
pytest tests/test_validation.py -v
```

| What is checked | Against what |
|---|---|
| Every Wigner 6j symbol | `sympy.physics.wigner`, exhaustively, all integer and half-integer arguments up to 2j = 3 — zero mismatches |
| θ and Δ | Factorial closed forms re-derived from scratch in the test file |
| 6j orthogonality | Evaluated *through* the `Sum()` machinery, so the summation bounds are on trial too |
| ‖closed θ net‖ | Θ(1,1,2)² = 900 |
| ‖tetrahedron‖ | θ(1,1,1)⁴ · W6j(1,1,1,1,1,1)² = 9216 |
| Node-naming invariance | The tetrahedron norm is identical under all 24 relabelings of its vertices |
| Backend agreement | Serial and parallel results are bitwise identical |
| **Transition sum rule** | **Σ_c P(c) = 1** over every admissible reconnection channel — see below |
| 6j canonicalisation | Symbols merged by the 24-fold tetrahedral key always have equal numerical value |

#### The transition sum rule

A reconnection merges two open legs *s* and *t* at a new trivalent vertex
carrying (*s*, *t*, *c*). Summing the transition probability over every
admissible channel *c* gives exactly **1** — the network has to go somewhere.

This is the strongest single check in the suite: it exercises the norms, the
Δ(c)/Θ(c,s,t) factor and the direction of the ratio simultaneously. Verified
for six parameter sets in `tests/test_probability.py::TestNormalisation`;
removing the Δ/Θ factor breaks 18 tests.

Run the whole suite with `pytest tests/`.

### Known Limitations

- **Trivalent only**: Every internal node must have exactly 3 edges. A graph that
  violates this now raises `ReductionError` rather than returning a partial product.
- **SU(2) only**: Classical (q → 1) symbols. No q-deformation, no quantum dimensions,
  no higher-valence intertwiners.
- **Planarity**: Face enumeration relies on a planar embedding. Non-planar graphs fall
  back to a minimum cycle basis computed on the *simple-graph* projection, which
  discards parallel edges — treat non-planar results with caution.
- **Evaluation cost is exponential** in the number of F-variables: the summation is a
  Cartesian product over their ranges, so cost grows as ∏ₖ(rangeₖ). Graph *reduction*
  is only polynomial; the summation is what limits problem size.
- **Conservative ranges**: When bounds cannot be derived symbolically, F-variables fall
  back to 0–20, which adds inadmissible terms that evaluate to zero but still cost time.
- **Memory**: Very large j (>1000) needs substantial RAM for wigxjpf tables (O(j²)).
- **No GPU path**: See [Computation Backends](#computation-backends). This is a
  property of wigxjpf, not a missing feature.

### Numerical Stability for Large Spins

The evaluator automatically handles large spin values (j up to 1000+) using:
- **Hybrid approach for theta**: Cached factorials for j ≤ 50, log-gamma for j > 50
- **Log-space for delta**: Always computes `(2j+1)^(2j)` as `exp(2j × log(2j+1))`
- **Vectorized operations**: Uses `scipy.special.gammaln` for efficient array computations
- **No overflow**: All factorial and power computations remain numerically stable

---

## License

GNU General Public License v3.0 (or later) — see [LICENSE](LICENSE).

---

## Authors

Nathan Cohen

**Contact:** nathan.cohen@univie.ac.at

---

## Acknowledgments

- **wigxjpf**: Developed by H. T. Johansson and C. Forssén at Chalmers University
- **NetworkX**: For graph data structures and algorithms
- **Matplotlib**: For graph visualization

---

## Further Reading

### Decomposition Theorem
- Moussouris, J. (1984), "Quantum models of space-time based on recoupling theory", Ph.D. thesis, University of Oxford.
- Ruiz, H.-C. (2012),"'Toroidal spin networks: Towards a generalization of the decomposition theorem"

### Spin Networks
- Penrose, R. (1971). "Angular momentum: an approach to combinatorial space-time"
- Kauffman, L. (1991). "Knots and Physics"

### Wigner Symbols
- Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). "Quantum Theory of Angular Momentum"
- Edmonds, A. R. (1996). "Angular Momentum in Quantum Mechanics"
