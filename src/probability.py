"""
Standalone probability calculator for spin network transitions.

This module provides calculate_probability(), a global utility that builds
the symbolic transition-probability *Formula* between any two SpinNetwork
states that share a valid ancestor-descendant relationship in the genealogy
tree.  This mirrors Graph.evaluate_symbolic(): the expensive graph-reduction
work happens once, up front, and the returned Formula is then evaluated
(possibly many times, possibly in a batch) exactly like any other Formula.

Formula
-------
    P = norm(G_out) x (Delta/Theta factor from reconnections)
        ------------------------------------------------------
              norm(G_in)  x  norm(G_Delta)

Where:
  - G_in  = parent network's graph
  - G_out = child network's graph
  - G_Delta = the added-edges subgraph (norm = 1 if no edges were added)
  - The Delta/Theta factor arises from each reconnection vertex: each
    reconnection triplet (c, s, t) contributes Delta(c) / Theta(c, s, t),
    and each consumed parent open end j contributes Delta(j).
  - Both denominators (norm(G_in) x norm(G_Delta)) and each Theta(c, s, t)
    are treated as "0 in, 0 out": a norm or Theta symbol that is exactly
    zero means the state/reconnection is physically forbidden, so the
    corresponding factor contributes 0 rather than raising a division error
    (see deltatheta() / safe_div() in spin_evaluator.py).

Usage
-----
    from src.api import load_network, calculate_probability

    n1 = load_network("drawn_graph.graphml")
    n2 = n1.transition_to()

    formula = calculate_probability(n1, n2)      # symbolic, like evaluate_symbolic()
    p = formula.evaluate_numeric()               # numeric value (args if any are free)
    probs = formula.evaluate_batch(args_list)     # scan many spin assignments efficiently
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from src.api import Formula, SpinNetwork


def calculate_probability(
    network_in: "SpinNetwork",
    network_out: "SpinNetwork",
) -> "Formula":
    """
    Build the symbolic transition-probability Formula for network_in -> network_out.

    Traces the path in the genealogy tree from network_in to network_out,
    composes intermediate Transitions for multi-hop paths, reduces the
    added-edges subgraph, and combines the three norms plus the
    reconnection Delta/Theta factor into a single Formula.

    This is the expensive step (equivalent to Graph.evaluate_symbolic()):
    it calls evaluate_symbolic() on network_in, network_out, and the added
    edges.  Each of those results is cached on its own Graph, so calling
    calculate_probability() again for the same pair is cheap.  Assigning
    spin values and evaluating is then done entirely on the returned Formula,
    exactly like any other Formula.

    Parameters
    ----------
    network_in : SpinNetwork
        The initial state (must be an ancestor of network_out).
    network_out : SpinNetwork
        The final state (must be a descendant of network_in).

    Returns
    -------
    Formula
        formula.get_args() lists every free spin variable coming from G_in,
        G_out, and G_delta combined.  Use set_args()/evaluate_numeric()/
        evaluate_batch() exactly as with any Formula returned by
        Graph.evaluate_symbolic().

    Raises
    ------
    LineageError
        If network_out is not a descendant of network_in.

    Examples
    --------
        formula = calculate_probability(n1, n2)
        p = formula.evaluate_numeric()                  # all labels numeric already

        args = formula.get_args()
        args[0].value = 1.5
        p = formula.evaluate_numeric(args)               # single custom assignment

        args_list = [[SpinArg("j_1", v)] for v in (0.5, 1.0, 1.5, 2.0)]
        probs = formula.evaluate_batch(args_list)         # scan efficiently
    """
    from src.api import Formula, Graph

    # --- Resolve the path, composing multi-hop transitions into one -----
    transitions = network_in.lineage_to(network_out)
    t = transitions[0]
    for tx in transitions[1:]:
        t = t.compose(tx)

    # --- Symbolic norms (each cached on its own Graph) -------------------
    formula_in = network_in.evaluate_symbolic()
    formula_out = network_out.evaluate_symbolic()

    added_nx = t.added_graph._nx_graph
    if added_nx.number_of_edges() > 0:
        formula_delta = Graph(added_nx).evaluate_symbolic()
        delta_str = formula_delta._formula_string
    else:
        delta_str = "1"

    # --- Delta/Theta factor from reconnection triplets and consumed ends -
    extra_str = _delta_theta_factor_string(t.theta_triplets, t.old_open_end_labels)

    # --- Combine into a single expression; safe_div avoids raising on a --
    # --- denominator that is exactly (and physically meaningfully) zero --
    combined = (
        f"safe_div(({formula_out._formula_string}) * ({extra_str}), "
        f"({formula_in._formula_string}) * ({delta_str}))"
    )

    return Formula._from_string(combined)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _label_literal(label) -> str:
    """
    Render an edge label (float, numeric string, or symbolic name/expression)
    as a Python-expression literal suitable for embedding into a formula
    string.  Symbolic names are sanitised the same way terms_to_formula_string
    does, so they line up with the SpinArg labels the resulting Formula
    exposes via get_args().
    """
    from src.spin_evaluator import _sanitize_primes

    if isinstance(label, (int, float)):
        return repr(float(label))
    text = str(label)
    try:
        return repr(float(text))
    except (TypeError, ValueError):
        return _sanitize_primes(text)


def _delta_theta_factor_string(
    theta_triplets: Tuple[Tuple, ...],
    old_open_end_labels: Tuple[float, ...],
) -> str:
    """
    Build the "Prod deltatheta(c,s,t) * Prod delta(j)" factor string.

    Parameters
    ----------
    theta_triplets : tuple of (c, s, t) tuples
        (new_label, old_label_1, old_label_2) for each reconnection.
    old_open_end_labels : tuple of float
        Labels of parent open ends consumed by this transition.

    Returns
    -------
    str
        "1" if there is nothing to multiply.
    """
    parts = [
        f"deltatheta({_label_literal(c)}, {_label_literal(s)}, {_label_literal(t)})"
        for c, s, t in theta_triplets
    ]
    parts += [f"delta({_label_literal(j)})" for j in old_open_end_labels]
    return " * ".join(parts) if parts else "1"
