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
```bash
# Workflow 1: Norm calculation
python scripts/graph.py              # Draw graph interactively
python scripts/compute_norm.py       # Symbolic reduction → PDFs
python scripts/evaluate_norm.py      # Numerical evaluation

# Workflow 2: Reconnection probabilities (GUI)
python scripts/transition_to.py drawn_graph.graphml

# Workflow 3: Probabilities (CLI)
python scripts/compute_all_probabilities.py drawn_graph.graphml 1-2 3-4
```

### Key Files by Purpose
| Purpose | File |
|---------|------|
| Graph editor | `scripts/graph.py` |
| Symbolic reduction | `src/graph_reducer.py` |
| Numerical eval | `src/spin_evaluator.py` |
| Canonicalization | `src/norm_reducer.py` |
| LaTeX PDFs | `src/LaTeX_rendering.py` |
| Graph gluing | `src/gluer.py` |
| Utilities | `src/utils.py` |

### Graph Editor Keys (scripts/graph.py)
`N` add node | `E` add edge | `M` move | `D` delete node | `X` delete edge | `Z` undo | `S` save

## Architecture

### Pipeline
```
Input Graph → Glue Open Edges → F-moves → Triangle Reductions → Expand 6j → Canonicalize → Evaluate
```

### Module Responsibilities
- **graph_reducer.py**: F-moves, triangle reductions, 6j symbol insertion
- **norm_reducer.py**: Kronecker constraints, Regge symmetries, canonicalization
- **spin_evaluator.py**: wigxjpf interface, JAX/multiprocessing backends, theta/delta symbols
- **gluer.py**: Creates theta graph by gluing open edges
- **utils.py**: Triangle inequality checks, face cycles, range computation

## Constraints

### Mathematical
- **Trivalent graphs only** (exactly 3 edges per node)
- **Triangle inequality**: |j₁-j₂| ≤ j₃ ≤ j₁+j₂ for edge labels
- **Half-integer spins**: 0, 1/2, 1, 3/2, 2, ...
- **F-variables**: Symbolic labels (F_1, F_2) for summation indices

### Technical
- Python 3.7+
- Core deps: networkx, matplotlib, sympy, numpy, scipy, pywigxjpf
- Optional: jax (GPU acceleration)
- Output files: `drawn_graph.graphml`, `norm_expression.pdf`, `canon_norm_expression.pdf`

## Testing
```bash
pytest tests/                        # All tests
pytest tests/test_integration.py     # Pipeline tests
pytest tests/test_graph_reducer.py   # Reduction tests
```

## Project Map
```
├── scripts/           # User-facing scripts
│   ├── graph.py                    # Interactive graph editor
│   ├── compute_norm.py             # Symbolic computation
│   ├── evaluate_norm.py            # Numerical evaluation
│   ├── compute_all_probabilities.py # Full probability distribution
│   ├── transition_to.py            # Reconnection GUI
│   ├── inspect_graph.py            # Graph inspection
│   └── modify_graph.py             # Graph modification
├── src/               # Core library
│   ├── graph_reducer.py            # F-moves, triangle reductions
│   ├── norm_reducer.py             # Canonicalization
│   ├── spin_evaluator.py           # Numerical evaluation
│   ├── gluer.py                    # Graph gluing
│   ├── LaTeX_rendering.py          # PDF generation
│   ├── utils.py                    # Utilities
│   ├── drawing.py                  # Visualization
│   └── reduction_animator.py       # Reduction GIFs
├── tests/             # Test suite
├── legacy/            # Deprecated (main.py)
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
- Use `--animate` flag with `compute_norm.py` for step visualization
- Check `graph_snapshots/` for intermediate states