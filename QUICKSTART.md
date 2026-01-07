# Quick Start Guide

**For collaborators who just want to compute spin network norms without understanding all the details.**

## Installation (One Time Only)

```bash
# 1. Navigate to the project folder
cd Spin_Networks_Project_full

# 2. Create virtual environment
python3 -m venv myenv

# 3. Activate it
source myenv/bin/activate  # On Mac/Linux
# OR
myenv\Scripts\activate     # On Windows

# 4. Install dependencies
pip install networkx matplotlib sympy pybind11 pywigxjpf
```

## Usage (Every Time)

### Option A: I want to draw a new spin network

```bash
# Activate environment
source myenv/bin/activate

# Draw your network
python scripts/graph.py
# → Click to add nodes
# → Right-click pairs to connect with edges
# → Enter spin values
# → Close window when done

# Compute symbolic form (generates PDFs)
python scripts/compute_norm.py

# Evaluate numerically
python scripts/evaluate_norm.py
```

### Option B: I have an existing GraphML file

```bash
# Activate environment
source myenv/bin/activate

# Compute and evaluate
python scripts/compute_norm.py my_network.graphml
python scripts/evaluate_norm.py my_network.graphml
```

## What You Get

### Files Generated:
- `norm_expression.pdf` - Raw symbolic expression
- `canon_norm_expression.pdf` - Simplified canonical form
- `reconstructed_canon_norm_expression.pdf` - Alternative representation

### Console Output:
```
======================================================================
✨ SPIN NETWORK NORM = -6.658558117818342e+01
======================================================================
```

This number is your result!

## Common Issues

### "File not found"
→ Run `python scripts/graph.py` first to create a network

### "Triangular condition not satisfied"
→ Your edge spins don't satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂ at some node

### Result is zero
→ Your spin network configuration is quantum mechanically forbidden

## Need More Help?

Read the full [README.md](README.md) for detailed explanations.
