# Graph Comparison and Modification Workflow

This workflow allows you to compare two spin network graphs and compute transition probabilities between them.

## Overview

The workflow consists of three main tools:

1. **`modify_graph.py`** - Interactive GUI for loading, modifying, and flagging edges in graphs
2. **`compare_graphs.py`** - Automated workflow that computes norms and transition data
3. **Flagging System** - Identifies open edges and extracts vertex information

## Key Concepts

### Open Edges
An **open edge** is an edge where at least one of its endpoints has degree < 3. These are the edges that connect to vertices that could accept additional edges.

### Flagging
When you flag an open edge, the system extracts:
- The edge nodes and label
- The vertex connected to the open end (degree < 3)
- The two other edges connected to that vertex
- Their labels for computing theta and delta coefficients

This information is used to compute transition probabilities between graphs.

## Usage

### Method 1: Manual Modification with `modify_graph.py`

Load and modify an existing graph interactively:

```bash
# Start with an existing graph
python scripts/modify_graph.py drawn_graph.graphml

# Or load a graph via File menu
python scripts/modify_graph.py
```

**Controls:**
- **N** - Add Node mode
- **E** - Add Edge mode (respects trivalent + triangular constraints)
- **M** - Move Node mode
- **D** - Delete Node mode
- **X** - Delete Edge mode
- **F** - Flag Edge mode (flag an open edge)
- **Z** - Undo
- **S** - Save & Exit

**Flagging an Edge:**
1. Press **F** to enter Flag mode
2. Open edges will appear highlighted in orange
3. Click on an open edge to flag it
4. The flagged edge will turn purple
5. Information will be displayed in the right panel
6. Save the graph - flagged data will be saved to `*_flagged.txt`

### Method 2: Automated Comparison with `compare_graphs.py`

Complete workflow from original graph to comparison results:

```bash
python scripts/compare_graphs.py drawn_graph.graphml
```

This will:
1. Compute the norm of the original graph
2. Open the GUI for you to modify and flag an edge
3. Compute the norm of the modified graph
4. Extract flagged edge information
5. Compute theta and delta coefficients
6. Display comparison results
7. Save everything to JSON file

**Options:**
```bash
# Specify output filename
python scripts/compare_graphs.py input.graphml -o output.graphml

# Skip GUI (use existing modified graph)
python scripts/compare_graphs.py input.graphml --skip-gui

# Quiet mode (less output)
python scripts/compare_graphs.py input.graphml -q
```

## Output Files

When you save a modified graph, several files are created:

1. **`modified_graph.graphml`** - The modified graph structure
2. **`modified_graph_flagged.txt`** - Human-readable flagged edge data
3. **`modified_graph_comparison.json`** - Complete comparison results

### Example JSON Output

```json
{
  "original_file": "drawn_graph.graphml",
  "modified_file": "modified_graph.graphml",
  "original_norm": -384.0,
  "modified_norm": -42.67,
  "norm_ratio": 0.111,
  "flagged_edge": {
    "edge_nodes": [2, 5],
    "edge_label": 1.0,
    "vertex_id": 5,
    "other_edge_labels": [1.0, 1.0]
  },
  "theta": -24.0,
  "delta": 3.0
}
```

## Transition Probability Formula

The probability of transitioning from graph G1 to graph G2 involves:

```
P(G1 → G2) ∝ N(G2) / N(G1) × f(θ, Δ)
```

Where:
- `N(G1)` = norm of original graph
- `N(G2)` = norm of modified graph
- `θ = theta(j1, j2, j3)` = theta symbol for the three edges at the flagged vertex
- `Δ = delta(j)` = delta symbol for the flagged edge
- `f(θ, Δ)` = some function of these coefficients (depends on your physics model)

The `compare_graphs.py` script computes all these quantities for you.

## Example Workflow

```bash
# 1. Create an initial graph (or use existing one)
python scripts/graph.py
# Draw your graph, save as drawn_graph.graphml

# 2. Run comparison workflow
python scripts/compare_graphs.py drawn_graph.graphml

# GUI opens:
#  - Add a new edge somewhere
#  - Flag one of the open edges (press F, click an orange edge)
#  - Save & Exit (press S)

# 3. View results
cat modified_graph_comparison.json

# 4. Repeat for different modifications
python scripts/compare_graphs.py drawn_graph.graphml -o variant2.graphml
```

## Constraints

The graph editor enforces spin network constraints:

1. **Trivalent constraint**: Each vertex can have at most 3 edges
2. **Triangular inequality**: For numeric spin labels j₁, j₂, j₃ at a vertex:
   ```
   |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂
   ```
3. **Half-integer spins**: Edge labels must be integers or half-integers (0, 0.5, 1, 1.5, 2, ...)

## Theta and Delta Symbols

The comparison script computes:

### Theta Symbol
```
θ(j₁, j₂, j₃) = (-1)^(j₁+j₂+j₃) × (j₁+j₂+j₃+1)! / [(j₁+j₂-j₃)! × (j₁-j₂+j₃)! × (-j₁+j₂+j₃)!]
```

For the three edges meeting at the flagged vertex.

### Delta Symbol
```
Δ(j) = (-1)^(2j) × (2j+1)
```

For the flagged edge label j.

## Tips

1. **Always flag before saving**: The comparison workflow needs flagged data
2. **One edge at a time**: Flag only one edge - the system extracts the relevant vertex info
3. **Open edges only**: You can only flag edges connected to vertices with degree < 3
4. **Undo is your friend**: Use Z to undo if you make a mistake
5. **Save often**: The GUI only saves on exit, so complete your modifications before exiting

## Troubleshooting

**Problem**: "Not an Open Edge" error when trying to flag
- **Solution**: Make sure the edge connects to a vertex with degree < 3

**Problem**: No flagged data in output
- **Solution**: Make sure you flagged an edge (F key) before saving

**Problem**: Triangular inequality violation
- **Solution**: Check that your edge labels satisfy |j₁-j₂| ≤ j₃ ≤ j₁+j₂ at each vertex

**Problem**: Can't add edge to vertex
- **Solution**: Vertices can have at most 3 edges (trivalent constraint)
