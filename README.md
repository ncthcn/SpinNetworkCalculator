# Spin Networks Calculator

A computational tool for calculating spin network norms and probabilities. This project performs symbolic graph reduction and numerical evaluation of spin networks using a combinatoric algorithm inspired by the Decomposition Theorem [1,2].

---

## Documentation Quick Links

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes (for collaborators)
- **[PROBABILITY_WORKFLOW.md](PROBABILITY_WORKFLOW.md)** - Compute reconnection probabilities
- **[PARALLEL_ACCELERATION.md](PARALLEL_ACCELERATION.md)** - GPU and parallel evaluation
- **[scripts/README_COMPARISON.md](scripts/README_COMPARISON.md)** - Graph comparison workflow
- **This README** - Comprehensive documentation

---

## Table of Contents

- [What is this?](#what-is-this)
- [Installation](#installation)
- [Quick Start](#quick-start)
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
   - Produces LaTeX PDFs and .txt files of the results

3. **Computes numerical values**:
   - Uses high-performance C++ backend (wigxjpf)
   - Handles large spin values efficiently
   - Returns the norm (positive scalar value) of the spin network


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
- SciPy for special functions (required for theta/delta symbols with large spins)
- JAX for GPU/parallel acceleration (optional but recommended)

**For Apple Silicon (M1/M2/M3) GPU acceleration:**
```bash
pip install jax-metal
```

**Alternative (manual installation):**
```bash
pip install networkx matplotlib sympy pybind11 pywigxjpf numpy scipy jax
```

### Step 2: Verify Installation

```bash
python -c "import pywigxjpf; print('✓ wigxjpf installed successfully')"
```

If you see the success message, you're ready to go.

---

## Quick Start

### 1. Draw Your Spin Network

Use the interactive graph drawing tool:

```bash
python scripts/graph.py
```

**Instructions (Keyboard-Driven Interface):**
- **Press N** - Add Node mode → Click anywhere to add nodes
- **Press E** - Add Edge mode → Click two nodes, then enter spin value
- **Press M** - Move Node mode → Click and drag nodes to reposition
- **Press D** - Delete Node mode → Click nodes to delete them
- **Press X** - Delete Edge mode → Click edges to delete them
- **Press Z** - Undo last action
- **Press S** - Save and exit

The graph is saved as `drawn_graph.graphml`

**Visual Feedback:**
- Current mode shown in right panel with color coding
- Selected nodes turn blue
- Dragging nodes turn orange
- Hover over edges to see them highlighted in red (delete mode)

### 2. Compute the Symbolic Norm

```bash
python scripts/compute_norm.py
```

**What it does:**
- Loads your graph
- Performs graph reduction (F-moves, triangle reductions)
- Generates canonical expression
- **Outputs:**
  - `norm_expression.pdf` (raw expression)
  - `canon_norm_expression.pdf` (simplified canonical form)

### 3. Evaluate Numerically

```bash
python scripts/evaluate_norm.py
```

**What it does:**
- Reads the canonical expression
- Computes Wigner 6j symbols using fast C++ backend
- **Uses parallel/GPU acceleration automatically** (JAX Metal on M3, or multiprocessing)
- **Outputs:** Numerical value of the spin network norm

**Example output:**
```
Using multiprocessing backend (11 workers)
Initializing wigxjpf tables for max 2j = 200...
✓ Wigxjpf initialized and ready

======================================================================
EVALUATING SPIN NETWORK EXPRESSION
======================================================================

Evaluating term 1/1...
  Computing summation over 3 variable(s)...
    Total iterations: 6,174
    Using 11 parallel workers
    Split into 44 chunks of ~140 iterations each
  Term value: 6.658558117818342e+01

======================================================================
 SPIN NETWORK NORM = 6.658558117818342e+01
======================================================================
```

---

## Usage Guide

### Reconnection Probability Workflow

**See [PROBABILITY_WORKFLOW.md](PROBABILITY_WORKFLOW.md) for complete documentation.**

Quick workflow (GUI):
1. Create graph with open edges: `python scripts/graph.py`
2. Launch reconnection GUI: `python scripts/transition_to.py drawn_graph.graphml`
3. In the GUI:
   - Click two open edges (orange)
   - Press C to connect
   - Choose "YES - All possible values" or "NO" and input specific value
   - Press S to save & compute

**Features:**
- **Visual selection** of which edges to reconnect
- **Automatic computation** of all possible edge values (through the triangle inequality)
- **Normalization test** verifies Σp = 1 (physical consistency)

The probability formula is:
```
p(c,...,z) = abs(Δ(c)⋅⋅⋅Δ(z) / [Θ(a,b,c)⋅⋅⋅Θ(x,y,z)] × ||G₂||/||G₁||)
```

Example output:
```
Probability distribution:
  New Edge     Probability
  0.0          0.000000e+00
  1.0          1.000000e+00
  2.0          0.000000e+00
  TOTAL        1.000000e+00

✓ PASSED: Probabilities sum to 1
  Physical consistency verified!
```

### Creating a Spin Network from Scratch

#### Method 1: Interactive Graph Editor (Recommended)

```bash
python scripts/graph.py
```

**The graph editor provides a modern, keyboard-driven interface:**

**Modes (switch with keyboard):**
1. **Add Node (N)**: Click anywhere on canvas to add nodes
2. **Add Edge (E)**: Click two nodes sequentially, enter spin value when prompted
3. **Move Node (M)**: Click and drag nodes to reposition them
4. **Delete Node (D)**: Click a node to delete it (and connected edges)
5. **Delete Edge (X)**: Click near an edge to delete it

**Additional Controls:**
- **Z** - Undo last action (up to 50 steps)
- **S** - Save graph and exit
- **Undo/Clear/Save buttons** - Available in top toolbar

**Interface Features:**
- **Right panel** shows current mode with instructions
- **Graph statistics** display node and edge counts
- **Edge curvature slider** adjusts parallel edge spacing
- **Hover feedback** - elements highlight when mouse hovers
- **Constraint validation** - Checks triangular inequality at nodes with 3 edges

**Tips:**
- Nodes can only have up to 3 edges (valence-3 constraint)
- Spin values must be integers or half-integers (0, 0.5, 1, 1.5, ...)
- Symbolic labels are supported for summation variables
- The editor validates triangular conditions: |j₁-j₂| ≤ j₃ ≤ j₁+j₂

#### Method 2: Use Existing GraphML File

If you already have a `.graphml` file:

```bash
python scripts/compute_norm.py my_network.graphml
python scripts/evaluate_norm.py my_network.graphml
```

### Advanced Options

#### Specify Maximum Spin Value

If you have very large spin values, you can manually set the maximum:

```bash
python scripts/evaluate_norm.py --max-j 50
```

This pre-allocates memory for spins up to j=50.

**For large spins (j up to 1000):**
```bash
python scripts/evaluate_norm.py --max-j 1000
```

The evaluator uses log-space computation to avoid numerical overflow for large spins:
- **Theta symbols**: Uses `scipy.special.gammaln` to compute factorials in log-space
- **Delta symbols**: Computes `(2j+1)^(2j)` as `exp(2j × log(2j+1))`
- **Memory**: Tables scale as O(j²), so j=1000 requires ~hundreds of MB

#### Quiet Mode (Less Output)

```bash
python scripts/evaluate_norm.py --quiet
```

Only shows the final result, useful for scripting.

---

## Understanding the Output

### 1. Console Output

When you run `compute_norm.py`, you'll see:

```
======================================================================
SPIN NETWORK NORM COMPUTATION (Symbolic)
======================================================================
Input file: drawn_graph.graphml

Loading graph...
  Loaded graph with 8 nodes and 8 edges
  ✓ Triangular condition satisfied for all nodes

Drawing original graph...
Gluing open edges (creating theta graph)...
  ✓ The glued graph is planar

Performing graph reduction (F-moves, triangle reductions)...
...
```

**Key checks:**
- **Triangular condition satisfied**: Each node's edges satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂
- **Graph is planar**: Can be drawn without crossing edges

If the glued graph is non-planar, `compute_norm.py` saves a Kuratowski obstruction image (`{input_basename}_kuratowski.png`, e.g. `drawn_graph_kuratowski.png`) showing the K₅ or K₃,₃ subdivision that witnesses non-planarity. Computation continues using the cycle-basis fallback unless `--strict-planarity` is passed.

### 2. PDF Output

#### `canon_norm_expression.pdf`
Shows the **canonical form** with:
- Combined duplicate coefficients
- Simplified signs
- Wigner 6j symbols
- Proper mathematical notation

### 3. TXT Output

#### `canon_norm_expression.txt`
Same content as `canon_norm_expression.pdf` expressed in a .txt file, ready to be given as input for `evaluate_formula.py`.

### 3. Numerical Result

The final number is the **norm** of your spin network state:

Example:
```
  SPIN NETWORK NORM = 6.658558117818342e+01
```

**Interpreting the result:**
- **Non-zero value**: Your spin network is physically allowed
- **Very small (~10⁻¹⁰)**: Might indicate numerical precision issues
- **Zero**: The spin network configuration violates SU(2) coupling rules

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
Canonicalize (combine duplicates, apply Regge symmetries)
    ↓
Numerical Evaluation (compute 6j values via wigxjpf)
    ↓
Final Result
```

---

## Troubleshooting

### Problem: "File not found"

**Error:**
```
Error: 'drawn_graph.graphml' not found.
```

**Solution:** First run `python scripts/graph.py` to create a spin network graph.

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

**Solution:** Reduce the `--max-j` parameter or use smaller spin values:
```bash
python scripts/evaluate_norm.py --max-j 100  # Limit to j ≤ 100
```

---

## File Structure

```
Spin_Networks_Project_full/
│
├── Documentation
│   ├── README.md                        # This file - comprehensive guide
│   ├── QUICKSTART.md                    # 5-minute quick start
│   ├── PROBABILITY_WORKFLOW.md          # Reconnection probability guide
│   ├── PARALLEL_ACCELERATION.md        # GPU/parallel evaluation guide
│   └── requirements.txt                 # Python dependencies
│
├── User Scripts (scripts/)
│   ├── graph.py                         # Interactive graph editor
│   ├── compute_norm.py                  # Symbolic computation (→ PDFs + .txt)
│   ├── evaluate_norm.py                 # Numerical evaluation
│   ├── compute_probability.py           # Single reconnection probability
│   ├── compute_all_probabilities.py     # Full probability distribution (CLI)
│   ├── compute_symbolic_probability.py  # Symbolic probability formula
│   ├── evaluate_formula.py              # Evaluate from canon_norm_expression.txt
│   ├── transition_to.py                 # Reconnection GUI
│   ├── compare_graphs.py                # Automated graph comparison workflow
│   ├── compare_graphs_cli.py            # Graph comparison (CLI)
│   ├── modify_graph.py                  # Interactive graph modification GUI
│   ├── inspect_graph.py                 # Graph inspection utility
│   └── README_COMPARISON.md            # Graph comparison workflow docs
│
├── Core Library (src/)
│   ├── drawing.py               # Graph visualization, Kuratowski plots
│   ├── gluer.py                 # Graph gluing operations
│   ├── graph_reducer.py         # F-moves and triangle reductions
│   ├── norm_reducer.py          # Canonicalization and Regge symmetries
│   ├── spin_evaluator.py        # Numerical evaluation with wigxjpf
│   ├── LaTeX_rendering.py       # PDF generation
│   ├── utils.py                 # Utility functions
│   ├── orientation.py           # Reference orientation calculations
│   └── reduction_animator.py   # Reduction step animation (GIFs)
│
├── Generated Files
│   ├── drawn_graph.graphml                      # User-drawn graph
│   ├── norm_expression.pdf                      # Raw symbolic expression
│   ├── canon_norm_expression.pdf                # Canonical expression (PDF)
│   ├── canon_norm_expression.txt                # Canonical expression (text)
│   ├── transition_to_graph.graphml              # Reconnected graph
│   ├── transition_to_graph_norm_G1.txt          # Original graph norm expression
│   ├── transition_to_graph_norm_G2.txt          # Reconnected graph norm expression
│   ├── transition_to_graph_symbolic_probability.txt  # Probability formula
│   ├── transition_to_graph_transition.json      # Full probability results
│   ├── graph_snapshots/graph.png                # Visualization snapshot
│   └── {input_basename}_kuratowski.png          # K₅/K₃,₃ subgraph (only if non-planar)
│
└── Other
    ├── tests/                   # Test suite (pytest)
    ├── legacy/                  # Deprecated (main.py)
    └── .gitignore               # Git ignore rules
```

---

## For Researchers

### Citation

The paper using this code and numerical results produced by it is in the writing process.
This section will be updated as soon as the paper is submitted.

---

### Known Limitations

- **Memory**: Very large j values (>1000) require substantial RAM for wigxjpf tables (scales as O(j²))
- **Planar graphs**: Non-planar graphs work but are slower
- **Conservative ranges**: Symbolic F-variables use conservative ranges (0-20) which may include extra iterations
- **6j with JAX**: Wigner 6j symbols still use C++ backend (pywigxjpf), not fully GPU-accelerated yet

### Numerical Stability for Large Spins

The evaluator automatically handles large spin values (j up to 1000+) using:
- **Hybrid approach for theta**: Cached factorials for j ≤ 100, log-gamma for j > 100
- **Log-space for delta**: Always computes `(2j+1)^(2j)` as `exp(2j × log(2j+1))`
- **Vectorized operations**: Uses `scipy.special.gammaln` for efficient array computations
- **No overflow**: All factorial and power computations remain numerically stable

---

## License

MIT License

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
