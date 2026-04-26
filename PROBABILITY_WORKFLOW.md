# Reconnection Probability Workflow

Compute transition probabilities for spin network edge reconnections with automatic normalization testing.

## Quick Start (Recommended)

```bash
# Step 1: Launch the GUI with your graph
python scripts/transition_to.py drawn_graph.graphml

# Step 2: In the GUI:
#   - Click two open edges (orange)
#   - Press C to connect
#   - Choose YES for "Compute all probabilities"
#   - Press Save & Compute
#
# Done! All probabilities computed automatically with normalization test.
```

## How It Works

### GUI Workflow

1. **Open your graph**:
   ```bash
   python scripts/transition_to.py drawn_graph.graphml
   ```

2. **Select two open edges**:
   - Open edges are highlighted in orange
   - Click two edges to select them (they turn red)

3. **Press Connect (C key)**:
   - A dialog appears asking: "Compute probabilities for:"
     - **YES - All possible values** (recommended for normalization test)
     - **NO - Single specific value**
     - **CANCEL - Abort reconnection**

4. **Choose your mode**:

   **Option A: YES - All Possible Values** (Recommended!)
   - Automatically computes ALL valid edge values from triangle inequality
   - Runs full probability calculation for each value
   - Verifies that Σp = 1 (normalization test)
   - No manual input needed!

   **Option B: NO - Single Value**
   - You enter one specific edge label
   - Creates reconnected graph
   - You manually run `compute_probability.py` later

5. **Press Save & Compute (S key)**:
   - If "All values": Computation runs automatically!
   - If "Single value": Graph saved for manual computation

### Example Session

```
$ python scripts/transition_to.py drawn_graph.graphml

[GUI opens]
[Click edge 1-2]
[Click edge 3-4]
[Press C]

Dialog: "Connecting edges with labels 7.5 and 9.5"
        "Possible new edge values: [2.0, 2.5, 3.0, ..., 17.0]"
        "Compute probabilities for:"
        "[YES - All possible values]"

[Click YES]
[Press S]

> Computing all probabilities...
> Reconnecting edges: 1-2 and 3-4
>
> Probability distribution:
>   New Edge     Probability
>   2.0          0.125
>   3.0          0.375
>   4.0          0.500
>   ...
>   TOTAL        1.000
>
> ✓ PASSED: Probabilities sum to 1
> Physical consistency verified!

[Done!]
```

## Mathematical Formula

The probability is computed as:

```
p(c,...,z) = abs(Δ(c)⋅⋅⋅Δ(z) / [Θ(a,b,c)⋅⋅⋅Θ(x,y,z)] × ||G₂||/||G₁||)
```

Where:
- **a, b**: Labels of the two edges being reconnected
- **c**: Label of the new open edge created
- **||G₁||**: Norm of original graph
- **||G₂||**: Norm of reconnected graph
- **Δ(c)**: Delta symbol = (-1)^(2c) × (2c+1)
- **Θ(a,b,c)**: Theta symbol for the reconnection vertex

## Normalization Test

**Critical requirement**: When summing over all possible values of c, the probabilities must equal 1:

```
Σ p(c) = 1
c
```

### Test Results

**Pass (✓)**:
```
Sum of all probabilities: 1.000000000000000e+00
✓ PASSED: Probabilities sum to 1 (within ε=1e-10)
  Physical consistency verified!
```
Your calculation is correct!

**Fail (✗)**:
```
Sum of all probabilities: 0.873452...
✗ FAILED: Probabilities do not sum to 1
```
Something is wrong - check for:
- Missing values in the sum
- Numerical precision errors
- Invalid graph structure
- Formula implementation bugs

## Triangle Inequality

The new edge label c must satisfy:

```
|a - b| ≤ c ≤ a + b
```

**Examples**:
- Edges 1.0 + 1.5 → possible values: [0.5, 1.5, 2.5]
- Edges 2.0 + 3.0 → possible values: [1.0, 2.0, 3.0, 4.0, 5.0]
- Edges 7.5 + 9.5 → possible values: [2.0, 3.0, ..., 17.0]

The GUI automatically determines all valid values.

## Command Line Usage

You can also use the command line directly (without GUI):

```bash
python scripts/compute_all_probabilities.py drawn_graph.graphml 1-2 3-4
```

Where `1-2` and `3-4` specify which edges to reconnect (format: `node1-node2`).

## Keyboard Shortcuts

In the GUI:
- **C** - Connect selected edges
- **Z** - Undo last action
- **S** - Save and compute

## Tips

1. **Always use "All values" mode** for physics calculations - the normalization test is crucial.

2. **Open edges** are those connected to vertices with degree < 3. They appear orange in the GUI.

3. **Large graphs may take time**: For graphs with high spin values, computation can take several minutes.

4. **Check the normalization**: If the test fails, your graph may have issues or the formula needs adjustment.

## What Gets Saved

When using `transition_to.py` (GUI workflow):
- `transition_to_graph.graphml` — reconnected graph structure
- `transition_to_graph_norm_G1.txt` — canonical norm expression for original graph
- `transition_to_graph_norm_G2.txt` — canonical norm expression for reconnected graph
- `transition_to_graph_symbolic_probability.txt` — symbolic probability formula
- `transition_to_graph_transition.json` — full results: norms, per-channel probabilities, normalization

When using `compute_all_probabilities.py` (CLI workflow):
- `*_all_probabilities.json` — full results with all probabilities and normalization test
- Console output — detailed step-by-step computation

## Troubleshooting

**"No open edges found"**:
- Your graph has no vertices with degree < 3
- All vertices are fully connected (trivalent)
- Add more edges or use a different graph

**"Probabilities don't sum to 1"**:
- Could be numerical precision (acceptable if close, e.g., 0.9999999)
- Could indicate invalid graph or formula error
- Check original graph norm isn't zero

**"Computation takes forever"**:
- High spin values require many 6j symbol calculations
- Try with smaller spin values first
- Use command line with `--quiet` flag to reduce output

## See Also

- [README.md](README.md) - Main documentation
- [QUICKSTART.md](QUICKSTART.md) - Getting started guide
- [scripts/README_COMPARISON.md](scripts/README_COMPARISON.md) - Graph comparison workflow
- [scripts/compute_probability.py](scripts/compute_probability.py) - Single value computation
- [scripts/compute_all_probabilities.py](scripts/compute_all_probabilities.py) - All values computation
- [scripts/compute_symbolic_probability.py](scripts/compute_symbolic_probability.py) - Symbolic probability
- [scripts/compare_graphs_cli.py](scripts/compare_graphs_cli.py) - CLI comparison tool
