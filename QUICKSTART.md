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
pip install -r requirements.txt

# Optional: For M1/M2/M3 Macs, add GPU acceleration
pip install jax-metal
```

**Note:** JAX provides automatic GPU/parallel acceleration. On M3 Macs with `jax-metal`, expect 10-100x speedup.

## Usage (Every Time)

### Option A: I want to draw a new spin network

```bash
# Activate environment
source myenv/bin/activate

# Draw your network
python scripts/graph.py
# → Press N to add nodes (click anywhere)
# → Press E to add edges (click two nodes, enter spin value)
# → Press M to move nodes (drag them)
# → Press D to delete nodes, X to delete edges
# → Press S to save and exit

# Compute symbolic form (generates PDFs and .txt)
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

### Files Generated

**From `compute_norm.py`:**
- `canon_norm_expression.pdf` — canonical expression (PDF)
- `canon_norm_expression.txt` — canonical expression (text, input for `evaluate_formula.py`)
- `norm_expression.pdf` — raw symbolic expression

**From `transition_to.py` (reconnection GUI):**
- `transition_to_graph.graphml` — reconnected graph
- `transition_to_graph_norm_G1.txt` — original graph norm expression
- `transition_to_graph_norm_G2.txt` — reconnected graph norm expression
- `transition_to_graph_symbolic_probability.txt` — probability formula
- `transition_to_graph_transition.json` — per-channel probabilities and norms

### Console Output:
```
🚀 Using multiprocessing backend (11 workers)
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
  Term value: -6.658558117818342e+01

======================================================================
✨ SPIN NETWORK NORM = -6.658558117818342e+01
======================================================================
```

This number is your result.

**Note:** The evaluation now runs in parallel automatically using all your CPU cores (or GPU if available).

## Common Issues

### "File not found"
→ Run `python scripts/graph.py` first to create a network

### "Triangular condition not satisfied"
→ Your edge spins don't satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂ at some node

### Result is zero
→ Your spin network configuration is forbidden by the rules of SU(2)

### Non-planar graph warning
→ `compute_norm.py` saves `{input_basename}_kuratowski.png` (e.g. `drawn_graph_kuratowski.png`) showing the K₅ or K₃,₃ subdivision witnessing non-planarity, then continues with cycle-basis fallback. Pass `--strict-planarity` to abort instead.

### "Memory error" with large spins
→ For spins up to j=1000, use:
```bash
python scripts/evaluate_norm.py --max-j 1000
```
The evaluator automatically uses log-space computation to avoid overflow.

## Advanced: Large Spin Values

The calculator handles large spins (j up to 1000+) automatically:
- **Theta symbols**: θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
  - Uses cached factorials for j ≤ 100 (fast)
  - Uses log-gamma for j > 100 (numerically stable)
- **Delta symbols**: Δⱼ = (-1)^(2j) × (2j+1)
  - Simple formula, no overflow issues

No special configuration needed - just specify `--max-j` for memory allocation!

## Need More Help?

- Full guide: [README.md](README.md)
- Probabilities: [PROBABILITY_WORKFLOW.md](PROBABILITY_WORKFLOW.md)
- GPU/parallel: [PARALLEL_ACCELERATION.md](PARALLEL_ACCELERATION.md)
- Graph comparison: [scripts/README_COMPARISON.md](scripts/README_COMPARISON.md)
