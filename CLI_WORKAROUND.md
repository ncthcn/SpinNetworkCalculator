# CLI Workaround for Tkinter Issues

## Problem

On some macOS systems, Tkinter has compatibility issues that cause crashes when trying to open the GUI. If you see an error like:

```
*** Terminating app due to uncaught exception 'NSInvalidArgumentException'
```

This is a known macOS/Tkinter compatibility issue, not a bug in the code.

## Solution: Use CLI Tools

I've created command-line alternatives that work without GUI:

### 1. Inspect Graph Tool

First, inspect your graph to identify open edges and get the flagging information:

```bash
python scripts/inspect_graph.py drawn_graph_with_labels.graphml
```

This shows:
- All open edges (edges connected to vertices with degree < 3)
- Degree of each vertex
- Which edges can be flagged

**To see details for a specific vertex:**

```bash
python scripts/inspect_graph.py drawn_graph_with_labels.graphml --vertex 2
```

This shows the exact command-line arguments you need for that vertex.

### 2. CLI Comparison Tool

Once you know which edge to flag, use the CLI comparison tool:

```bash
python scripts/compare_graphs_cli.py original.graphml modified.graphml \
    --flagged-edge 2 1 \
    --flagged-label 1.0 \
    --flagged-vertex 2 \
    --other-labels 1.0 1.0
```

This computes:
- Original norm
- Modified norm
- Norm ratio
- Theta and delta coefficients
- Saves everything to JSON

## Complete Workflow Example

```bash
# Step 1: Create your graphs manually (or use existing ones)
# You can still use graph.py if it doesn't crash, or create .graphml files manually

# Step 2: Inspect the original graph
python scripts/inspect_graph.py my_graph.graphml

# Step 3: Look at a specific vertex (e.g., vertex 2)
python scripts/inspect_graph.py my_graph.graphml --vertex 2
# This shows you the exact --flagged-edge, --flagged-label, etc. arguments

# Step 4: Create/modify your second graph
# (manually edit .graphml or use graph.py if it works)

# Step 5: Run comparison
python scripts/compare_graphs_cli.py original.graphml modified.graphml \
    --flagged-edge 2 1 \
    --flagged-label 1.0 \
    --flagged-vertex 2 \
    --other-labels 1.0 1.0 \
    --output my_comparison.json

# Step 6: View results
cat my_comparison.json
```

## Example: Comparing the Triangle Graph

Using your existing `drawn_graph_with_labels.graphml`:

```bash
# 1. Inspect it to see open edges
python scripts/inspect_graph.py drawn_graph_with_labels.graphml --vertex 2

# This tells you vertex 2 has three edges with labels [1.0, 1.0, 1.0]
# Let's flag the edge from vertex 2 to node 1

# 2. Create a modified version (manually edit or use graph.py)
# For this example, let's say you manually edited it to modified_triangle.graphml

# 3. Run comparison
python scripts/compare_graphs_cli.py \
    drawn_graph_with_labels.graphml \
    modified_triangle.graphml \
    --flagged-edge 2 1 \
    --flagged-label 1.0 \
    --flagged-vertex 2 \
    --other-labels 1.0 1.0

# Output:
# Original norm: -384.0
# Modified norm: <computed>
# Theta(1, 1, 1) = -24
# Delta(1) = 3
```

## If You Don't Need Flagging

If you just want to compare norms without flagging edges:

```bash
python scripts/compare_graphs_cli.py original.graphml modified.graphml
```

This skips the theta/delta calculations but still gives you the norm ratio.

## Manually Creating/Editing GraphML Files

If even `graph.py` crashes, you can manually create `.graphml` files. Here's a minimal template:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<graphml>
  <key id="label" for="edge" attr.name="label" attr.type="double"/>
  <key id="x" for="node" attr.name="x" attr.type="double"/>
  <key id="y" for="node" attr.name="y" attr.type="double"/>
  <graph edgedefault="undirected">
    <node id="1">
      <data key="x">100.0</data>
      <data key="y">100.0</data>
    </node>
    <node id="2">
      <data key="x">200.0</data>
      <data key="y">100.0</data>
    </node>
    <edge source="1" target="2">
      <data key="label">1.0</data>
    </edge>
  </graph>
</graphml>
```

## Fixing Tkinter (Optional)

If you want to fix the Tkinter issue, try:

```bash
# Reinstall Python with proper Tk support
brew reinstall python-tk@3.11

# Or use Python from python.org instead of Homebrew
```

But the CLI tools work fine without fixing Tkinter.

## Tools Summary

| Tool | Purpose | Requires GUI? |
|------|---------|---------------|
| `graph.py` | Create graphs visually | Yes ✗ |
| `modify_graph.py` | Modify + flag graphs visually | Yes ✗ |
| `compare_graphs.py` | Full workflow with GUI | Yes ✗ |
| **`inspect_graph.py`** | **Find open edges, get flagging info** | **No ✓** |
| **`compare_graphs_cli.py`** | **Complete comparison without GUI** | **No ✓** |

The bolded tools work on all systems, including those with Tkinter issues.

## Getting Help

```bash
# Inspect tool help
python scripts/inspect_graph.py --help

# CLI comparison help
python scripts/compare_graphs_cli.py --help
```

## Testing

Test the CLI workflow:

```bash
# This should work even with Tkinter issues
python scripts/test_comparison.py
```

All tests should pass, confirming the workflow works.
