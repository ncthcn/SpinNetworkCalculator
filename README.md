# Spin Networks Calculator

A computational tool for calculating spin network norms in loop quantum gravity. This project performs symbolic graph reduction and numerical evaluation of spin networks using Wigner 6j symbols.

---

## 📂 Documentation Quick Links

- 🚀 **[QUICKSTART.md](QUICKSTART.md)** - Get started in 5 minutes (for collaborators)
- 📖 **This README** - Comprehensive documentation (you're reading it!)

---

## 📋 Table of Contents

- [What is this?](#what-is-this)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Understanding the Output](#understanding-the-output)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## 🎯 What is this?

**Spin networks** are mathematical structures used in loop quantum gravity to describe quantum states of space. This tool:

1. **Takes a graph as input** where:
   - Nodes represent quantum states
   - Edges have numerical labels (spin values)

2. **Performs symbolic computation**:
   - Applies F-moves and triangle reductions
   - Generates canonical expressions with Wigner 6j symbols
   - Produces beautiful LaTeX PDFs of the results

3. **Computes numerical values**:
   - Uses high-performance C++ backend (wigxjpf)
   - Handles large spin values efficiently
   - Returns the norm (scalar value) of the spin network

---

## 🚀 Installation

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

**Alternative (manual installation):**
```bash
pip install networkx matplotlib sympy pybind11 pywigxjpf
```

### Step 2: Verify Installation

```bash
python -c "import pywigxjpf; print('✓ wigxjpf installed successfully')"
```

If you see the success message, you're ready to go! 🎉

---

## ⚡ Quick Start

### 1. Draw Your Spin Network

Use the interactive graph drawing tool:

```bash
python scripts/graph.py
```

**Instructions:**
- **Left click** to add nodes
- **Right click** on two nodes to connect them with an edge
- Enter the spin value (numerical label) for each edge
- **Close the window** when done
- The graph is saved as `drawn_graph_with_labels.graphml`

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
- **Outputs:** Numerical value of the spin network norm

**Example output:**
```
======================================================================
✨ SPIN NETWORK NORM = -6.658558117818342e+01
======================================================================
```

---

## 📖 Usage Guide

### Creating a Spin Network from Scratch

#### Method 1: Interactive Drawing (Recommended for Beginners)

```bash
python scripts/graph.py
```

1. Click to place nodes
2. Right-click on pairs of nodes to create edges
3. Enter spin values when prompted
4. Close the window to save

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

#### Quiet Mode (Less Output)

```bash
python scripts/evaluate_norm.py --quiet
```

Only shows the final result, useful for scripting.

---

## 📊 Understanding the Output

### 1. Console Output

When you run `compute_norm.py`, you'll see:

```
======================================================================
SPIN NETWORK NORM COMPUTATION (Symbolic)
======================================================================
Input file: drawn_graph_with_labels.graphml

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
- ✓ **Triangular condition satisfied**: Each node's edges satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂
- ✓ **Graph is planar**: Can be drawn without crossing edges (important for physical realizability)

### 2. PDF Outputs

#### `norm_expression.pdf`
Shows the **raw expression** after graph reduction:
```
∑[F₁=2,...,3] × 6j(...) × 6j(...) × θ(...) × Δ(...)
```

#### `canon_norm_expression.pdf`
Shows the **canonical form** with:
- Combined duplicate coefficients
- Simplified signs
- W6j symbols (weighted Wigner 6j)
- Proper mathematical notation

### 3. Numerical Result

The final number is the **norm** (magnitude) of your spin network state:

```
✨ SPIN NETWORK NORM = -6.658558117818342e+01
```

**Interpreting the result:**
- **Non-zero value**: Your spin network is physically allowed ✓
- **Very small (~10⁻¹⁰)**: Might indicate numerical precision issues
- **Zero**: The spin network configuration violates quantum coupling rules

---

## 🔬 Technical Details

### Mathematical Background

A **spin network** is a graph with:
- Edges labeled by spins (half-integers: 0, 1/2, 1, 3/2, 2, ...)
- Nodes satisfying the **triangle inequality**: For edges j₁, j₂, j₃ meeting at a node:
  ```
  |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
  ```

The **norm** is computed as a product of:
- **Wigner 6j symbols**: Quantum recoupling coefficients
- **Theta symbols**: θ(j₁,j₂,j₃) = √[(2j₁+1)(2j₂+1)(2j₃+1)]
- **Delta symbols**: Δⱼ = √(2j+1)
- **Sign factors**: (-1)^(...)

### Algorithm Overview

```
Input Graph
    ↓
Glue Open Edges (create theta graph)
    ↓
Apply F-moves (insert 6j symbols)
    ↓
Triangle Reductions (simplify graph)
    ↓
Expand 6j → W6j (with theta/delta factors)
    ↓
Canonicalize (combine duplicates, apply Regge symmetries)
    ↓
Numerical Evaluation (compute 6j values via wigxjpf)
    ↓
Final Result
```

### Why wigxjpf?

The **wigxjpf** library (Wigner symbols using prime factorization):
- ✅ Avoids factorial overflow (uses prime factorization)
- ✅ Industry standard (extensively tested)
- ✅ Handles half-integer spins natively (stores as 2×j)
- ✅ Much faster than naive implementations (~1000× speedup)

---

## 🔧 Troubleshooting

### Problem: "File not found"

**Error:**
```
Error: 'drawn_graph_with_labels.graphml' not found.
```

**Solution:** First run `python scripts/graph.py` to create a spin network graph.

---

### Problem: "Triangular condition not satisfied"

**Error:**
```
✗ Error: Triangular condition not satisfied at node X
```

**Solution:** The spin values at node X violate the triangle inequality. For edges j₁, j₂, j₃:
- Check that |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
- Example: Edges with spins 1, 5, 10 violate this because |1-5| = 4 but 10 > 1+5 = 6

---

### Problem: Result is exactly zero

**Symptom:**
```
✨ SPIN NETWORK NORM = 0.000000000000000e+00
```

**Possible causes:**
1. **Quantum forbidden configuration**: Some 6j symbols have zero value because internal triangle conditions aren't satisfied
2. **Cancellation**: Multiple terms with opposite signs sum to zero

**What to try:**
- Check that your graph edges satisfy triangular conditions at each node
- Try different spin values
- Verify the graph structure is physically meaningful

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

## 📚 File Structure

```
Spin_Networks_Project_full/
│
├── 📄 Documentation
│   ├── README.md                # This file - comprehensive guide
│   ├── QUICKSTART.md            # 5-minute quick start
│   └── requirements.txt         # Python dependencies
│
├── 📜 User Scripts (scripts/)
│   ├── graph.py                 # Interactive graph drawing tool
│   ├── compute_norm.py          # Symbolic computation (→ PDFs)
│   └── evaluate_norm.py         # Numerical evaluation (→ result)
│
├── 🔧 Core Library (src/)
│   ├── drawing.py               # Graph visualization utilities
│   ├── gluer.py                 # Graph gluing operations
│   ├── graph_reducer.py         # F-moves and triangle reductions
│   ├── norm_reducer.py          # Canonicalization and Regge symmetries
│   ├── spin_evaluator.py        # Numerical evaluation with wigxjpf
│   ├── LaTeX_rendering.py       # PDF generation
│   └── utils.py                 # Utility functions
│
├── 📊 Generated Files (ignored by git)
│   ├── drawn_graph_with_labels.graphml
│   ├── norm_expression.pdf
│   ├── canon_norm_expression.pdf
│   └── reconstructed_canon_norm_expression.pdf
│
└── 🔧 Other
    ├── main.py                  # Legacy combined pipeline
    └── .gitignore               # Git ignore rules
```

---

## 🎓 For Researchers

### Citation

If you use this tool in your research, please cite:

```
[Your paper/thesis information here]
```

### Understanding the Physics

The spin network norm ⟨ψ|ψ⟩ represents the inner product of a spin network state with itself. Key physical interpretations:

- **Norm = 0**: State is orthogonal to itself → unphysical
- **Norm > 0**: Properly normalized quantum state
- **Sign**: Can be negative due to phase conventions in Wigner 6j symbols

### Validation

This code has been validated against:
- Hand calculations for simple graphs (tetrahedron, etc.)
- Mathematica implementations of Wigner 6j symbols
- Known results from loop quantum gravity literature

---

## 🤝 Contributing

Found a bug? Have a feature request?

1. Check existing issues at [your repository]
2. Create a detailed bug report with:
   - Input graph (GraphML file)
   - Expected vs. actual output
   - Python version and OS

---

## 🚧 Future Work

### Features to be Implemented

- [ ] **Multiple summation variables**: Currently limited to single F variable summation
- [ ] **Parallel evaluation**: Evaluate independent terms concurrently for speed
- [ ] **Caching of 6j values**: Store computed Wigner symbols to avoid recomputation
- [ ] **GPU acceleration**: Use CUDA for large j values
- [ ] **Web interface**: Browser-based graph drawing tool
- [ ] **Batch processing**: Process multiple graphs in one command
- [ ] **Export formats**: Mathematica, Maple, JSON output
- [ ] **Visualization**: Animate reduction steps
- [ ] **Unit tests**: Comprehensive test suite

### Known Limitations

- **Single summation**: Can only handle one summation variable (F_1, F_2, etc.)
- **Memory**: Large j values (>100) require substantial RAM for wigxjpf tables
- **Planar graphs**: Non-planar graphs work but are slower
- **Debug output**: Some internal print statements clutter output (can be suppressed)

### Contributing

If you'd like to contribute:
1. Pick an item from "Future Work" above
2. Fork the repository
3. Create a feature branch
4. Submit a pull request with tests

---

## 📄 License

[Your license here]

---

## 👥 Authors

[Your name and collaborators]

**Contact:** [Your email]

---

## 🙏 Acknowledgments

- **wigxjpf**: Developed by H. T. Johansson and C. Forssén at Chalmers University
- **NetworkX**: For graph data structures and algorithms
- **Matplotlib**: For graph visualization

---

## 📖 Further Reading

### Loop Quantum Gravity
- Rovelli, C., & Vidotto, F. (2014). *Covariant Loop Quantum Gravity*
- Thiemann, T. (2007). *Modern Canonical Quantum General Relativity*

### Spin Networks
- Penrose, R. (1971). "Angular momentum: an approach to combinatorial space-time"
- Kauffman, L. (1991). *Knots and Physics*

### Wigner Symbols
- Varshalovich, D. A., Moskalev, A. N., & Khersonskii, V. K. (1988). *Quantum Theory of Angular Momentum*
- Edmonds, A. R. (1996). *Angular Momentum in Quantum Mechanics*

---

**Happy Computing!** 🚀✨
