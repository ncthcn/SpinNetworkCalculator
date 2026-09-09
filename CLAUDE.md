# For Claude

## Output Constraints (Mandatory)
- **Plan output**: Maximum 200 words
- **Heading hierarchy**: Up to h3 only (h4+ prohibited)
- **Specificity focus**: Include file names, commands, line numbers
- No verbose explanations
- Prohibit abstract expressions like "detailed analysis", "comprehensive review"
- Code examples limited to 10 lines maximum
- **Next action**: End with 1-3 specific questions

## Suggestions Format
All suggestions must follow this format:

**What**: [Specific file or command]
**Where**: [File path:line number]
**Why**: [Reason within 20 characters]

# Spin Networks Calculator

Computational tool for spin network norms using symbolic graph reduction and Wigner 6j symbols. Implements Penrose spin networks theory for quantum gravity/angular momentum calculations.

## Quick Reference

### Primary Workflows
```python
# Workflow 1: Norm calculation (library API)
from src.api import new_network, load_network
snet = new_network()                      # opens drawing GUI
formula = snet.evaluate_symbolic()        # full reduction pipeline
formula.save("result.pdf", "pdf")         # or "result.txt", "txt"
result = formula.evaluate_numeric()       # numerical value

# Workflow 2: Norm from saved expression
from src.api import Formula
formula = Formula.load("canon_norm_expression.txt")
result = formula.evaluate_numeric()

# Workflow 3: Transition (GUI produces child network, no computation)
n2 = n1.transition_to()                   # opens GUI, returns child SpinNetwork

# Workflow 4: Transition probability (API)
from src.api import calculate_probability
formula = calculate_probability(n1, n2)   # symbolic, like evaluate_symbolic()
p = formula.evaluate_numeric()            # numeric value
probs = formula.evaluate_batch(args_list) # scan many spin assignments

# Workflow 5: Graph comparison
```

```bash
python scripts/compare_graphs.py drawn_graph.graphml
```

### Key Files by Purpose
| Purpose | File |
|---------|------|
| Public library API | `src/api.py` |
| Genealogy / transitions | `src/evolution.py` |
| Probability calculation | `src/probability.py` |
| Tree visualization | `src/visualizer.py` |
| Graph editor | `scripts/graph.py` |
| Symbolic reduction | `src/graph_reducer.py` |
| Numerical eval | `src/spin_evaluator.py` |
| Canonicalization | `src/norm_reducer.py` |
| LaTeX PDFs | `src/LaTeX_rendering.py` |
| Graph gluing | `src/gluer.py` |
| Utilities | `src/utils.py` |
| Orientations | `src/orientation.py` |
| Graph comparison | `scripts/compare_graphs.py` |

### Graph Editor Keys (scripts/graph.py)
`N` add node | `E` add edge | `M` move | `D` delete node | `X` delete edge | `Z` undo | `S` save

## Architecture

### Pipeline
```
Input Graph → Glue Open Edges → F-moves → Triangle Reductions → Expand 6j → Canonicalize → Evaluate

Reduction is polynomial; the final summation is exponential in the number of
F-variables. Failure to fully reduce raises ReductionError (never a partial result).
abs() is applied once, at the api.py boundary, not inside the evaluator.
```

### Module Responsibilities
- **api.py**: Public library entry point; `SpinNetwork`, `Graph`, `Formula`, `SpinArg` classes; `new_network()`, `load_network()` factory functions
- **evolution.py**: `Transition` (genealogy edge), `LineageError`; `Transition.compose()` for multi-hop paths
- **probability.py**: `calculate_probability(n_in, n_out) -> Formula` standalone function
- **visualizer.py**: `TreeVisualizer.display_tree()`, `ascii_tree()` for genealogy plots
- **graph_reducer.py**: F-moves, triangle reductions, 6j symbol insertion
- **norm_reducer.py**: Kronecker constraints, 24-fold tetrahedral 6j symmetry, canonicalization
- **spin_evaluator.py**: wigxjpf interface, serial/multiprocessing backends, theta/delta symbols
- **gluer.py**: Creates theta graph by gluing open edges
- **utils.py**: Triangle inequality checks, face cycles, range computation
- **orientation.py**: Reference orientation and layout phase calculations
- **drawing.py**: Graph visualization, Kuratowski obstruction plots

## Constraints

### Mathematical
- **Trivalent graphs only** (exactly 3 edges per node)
- **Triangle inequality**: |j₁-j₂| ≤ j₃ ≤ j₁+j₂ for edge labels
- **Half-integer spins**: 0, 1/2, 1, 3/2, 2, ...
- **F-variables**: Symbolic labels (F_1, F_2) for summation indices

### Technical
- Python 3.7+
- Core deps: networkx, matplotlib, sympy, numpy, scipy, pywigxjpf
- Output files:
  - `drawn_graph.graphml` — user-drawn graph (via `graph.save()`)
  - `norm_expression.pdf` — raw symbolic expression
  - `canon_norm_expression.pdf` — canonical expression (PDF)
  - `canon_norm_expression.txt` — canonical expression (text, reload via `Formula.load()`)
  - `transition_to_graph.graphml` — child graph from `transition_to()` GUI
  - `transition_to_graph_transition.json` — structural metadata (added edges, reconnections)
  - `graph_snapshots/graph.png` — visualization snapshot
  - `{input_basename}_kuratowski.png` — K₅/K₃,₃ obstruction subgraph (non-planar graphs only)

## Testing
```bash
pytest tests/                        # All tests
pytest tests/test_integration.py     # Pipeline tests
pytest tests/test_graph_reducer.py   # Reduction tests
pytest tests/test_validation.py      # Validation vs sympy / closed forms / known norms
pytest tests/test_symbols.py         # Wigner symbol tests
pytest tests/test_orientation.py     # Orientation tests
pytest tests/test_reconnection_workflow.py  # Reconnection probability tests
# NOTE: tests/test_multi_sum.py defines no test functions; it runs at import
# time and writes test_triple_sum.pdf into the repo root. It is a demo script,
# not a test.
pytest tests/test_ranges.py          # Range calculation tests
```

## Project Map
```
├── scripts/           # User-facing scripts
│   ├── graph.py                         # Interactive graph editor
│   ├── transition_to.py                 # Transition GUI (produces child graph)
│   ├── compute_probability.py           # Single reconnection probability (CLI)
│   ├── compute_all_probabilities.py     # Full probability distribution (CLI)
│   ├── compute_symbolic_probability.py  # Symbolic probability formula (CLI)
│   ├── check_backends.py                # Verifies serial == parallel
│   ├── benchmark_backends.py            # Measures the parallel break-even point
│   ├── compare_graphs.py                # Automated graph comparison workflow
│   ├── compare_graphs_cli.py            # Graph comparison (CLI)
│   ├── inspect_graph.py                 # Graph inspection
│   ├── modify_graph.py                  # Interactive graph modification GUI
│   └── README_COMPARISON.md             # Graph comparison workflow docs
├── src/               # Core library
│   ├── api.py                      # Public API (SpinNetwork, Graph, Formula, SpinArg)
│   ├── evolution.py                # Transition class, LineageError
│   ├── probability.py              # calculate_probability()
│   ├── visualizer.py               # TreeVisualizer
│   ├── graph_reducer.py            # F-moves, triangle reductions
│   ├── norm_reducer.py             # Canonicalization
│   ├── spin_evaluator.py           # Numerical evaluation
│   ├── gluer.py                    # Graph gluing
│   ├── LaTeX_rendering.py          # PDF generation
│   ├── utils.py                    # Utilities
│   ├── drawing.py                  # Visualization
│   ├── orientation.py              # Reference orientation calculations
│   └── reduction_animator.py       # Reduction GIFs
├── tests/             # Test suite
│   ├── test_graph_reducer.py
│   ├── test_integration.py
│   ├── test_multi_sum.py                # (demo script; defines no tests)
│   ├── test_validation.py               # Independent numerical validation
│   ├── test_orientation.py
│   ├── test_range_improvements.py
│   ├── test_ranges.py
│   ├── test_reconnection_workflow.py
│   └── test_symbols.py
└── graph_snapshots/   # Generated images
```

## Common Tasks

### Adding new graph operations
1. Core logic in `src/` modules
2. User interface in `scripts/`
3. Add tests in `tests/`

### Modifying evaluation
- Backend selection: `spin_evaluator.py` → `SpinNetworkEvaluator`
- Symbol computation: `spin_evaluator.py` → `compute_theta()`, `compute_delta()`
- 6j symbols: Uses pywigxjpf C++ backend

### Debugging reductions
- Check `graph_snapshots/` for intermediate states
