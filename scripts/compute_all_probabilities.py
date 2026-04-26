#!/usr/bin/env python3
"""
Compute All Reconnection Probabilities

Computes probabilities for ALL possible new edge values when reconnecting two open edges,
and verifies that they sum to 1 (normalization test).

This is useful for:
1. Verifying physical consistency (probabilities must sum to 1)
2. Finding the most probable transition
3. Understanding the full probability distribution

Usage:
    python compute_all_probabilities.py original.graphml edge1 edge2

Where edge1 and edge2 are specified as "node1-node2" (e.g., "1-2")
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import networkx as nx

from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.norm_reducer import canonicalise_terms, apply_kroneckers, expand_6j_symbolic
from src.spin_evaluator import evaluate_spin_network, SpinNetworkEvaluator
from src.LaTeX_rendering import terms_to_formula_string


# Standard graph loader shared by all scripts: reads GraphML, coerces labels
# to float, populates "pos" from stored x/y or kamada-kawai fallback.
def load_graph_from_file(file_path):
    """Load a graph from a GraphML file and ensure all nodes have valid positions."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    # Convert edge labels to float or keep as string
    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except (ValueError, KeyError):
            pass  # Keep non-numeric labels as strings

    # Ensure all nodes have valid positions
    pos = nx.kamada_kawai_layout(graph)
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            graph.nodes[node]["pos"] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        else:
            graph.nodes[node]["pos"] = pos[node]

    return graph


# Thin wrapper: loads the file then delegates to compute_norm_from_graph.
def compute_norm_from_file(graph_file, quiet=True):
    """Compute spin network norm from a GraphML file."""
    if not quiet:
        print(f"Computing norm for {graph_file}...")

    graph = load_graph_from_file(graph_file)
    return compute_norm_from_graph(graph, quiet=quiet)


# Core norm pipeline: glue → reduce → Kronecker → expand 6j → canonicalise
# → evaluate. Returns a float. The same logic appears in evaluate_norm.py;
# having it here avoids subprocess calls and lets batch loops stay in-process.
def compute_norm_from_graph(graph, quiet=True):
    """Compute spin network norm from a NetworkX graph object."""
    canon_terms, result = _compute_norm_full(graph)
    if not quiet:
        print(f"  → Norm = {result}")
    return result


def _compute_norm_full(graph):
    """Run full norm pipeline; return (canon_terms, norm_float)."""
    glued_graph = glue_open_edges(graph)
    terms = reduce_all_cycles(glued_graph, animator=None)

    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)

    for term in clean_terms:
        new = []
        for c in term["coeffs"]:
            if isinstance(c, dict) and c.get("type") == "6j":
                expanded = expand_6j_symbolic(c)
                new.extend(expanded)
            else:
                new.append(c)
        term["coeffs"] = new

    canon_terms = canonicalise_terms(clean_terms)
    result = evaluate_spin_network(canon_terms)
    return canon_terms, result


# Short aliases used inside main() for readability.
def load_graph(file_path):
    return load_graph_from_file(file_path)


def compute_norm(graph_file, quiet=True):
    return compute_norm_from_file(graph_file, quiet=quiet)


# Enumerates all c values satisfying the triangle inequality |j1-j2| ≤ c ≤ j1+j2
# with integer step (the integer-sum rule requires j1+j2+c ∈ Z, so step is always 1).
def calculate_possible_values(j1, j2):
    """Calculate all possible values for new edge based on triangle inequality."""
    possible = []
    j_min = abs(j1 - j2)
    j_max = j1 + j2

    # Determine step size (1.0 for integer, 0.5 for half-integer)
    if (j1 % 1 == 0) and (j2 % 1 == 0):
        step = 1.0
    else:
        step = 0.5

    current = j_min
    while current <= j_max:
        possible.append(current)
        current += step

    return possible


# Builds the reconnected graph G2 for a given new label c:
# removes the two specified open edges, inserts a new trivalent vertex,
# and adds a new open edge with label new_label. Returns (G2, label1, label2).
# Edge specs are "node1-node2" strings; the open endpoint is auto-detected as
# the one with degree < 3.
def reconnect_edges_with_label(graph, edge1_spec, edge2_spec, new_label):
    graph = graph.copy()

    # Parse edge specifications
    n1_1, n1_2 = edge1_spec.split('-')
    n2_1, n2_2 = edge2_spec.split('-')

    # Find the edges in the graph
    edge1_data = None
    edge2_data = None

    for u, v, key, data in graph.edges(keys=True, data=True):
        if (u == n1_1 and v == n1_2) or (u == n1_2 and v == n1_1):
            edge1_data = (u, v, key, data)
        if (u == n2_1 and v == n2_2) or (u == n2_2 and v == n2_1):
            edge2_data = (u, v, key, data)

    if edge1_data is None or edge2_data is None:
        raise ValueError(f"Could not find edges {edge1_spec} or {edge2_spec}")

    u1, v1, k1, data1 = edge1_data
    u2, v2, k2, data2 = edge2_data

    label1 = data1.get('label', 1.0)
    label2 = data2.get('label', 1.0)

    # Determine which endpoints are open (degree < 3)
    open1 = u1 if graph.degree(u1) < 3 else v1
    other1 = v1 if open1 == u1 else u1

    open2 = u2 if graph.degree(u2) < 3 else v2
    other2 = v2 if open2 == u2 else u2

    # Create new reconnection node
    new_node = max(int(n) for n in graph.nodes()) + 1
    graph.add_node(str(new_node))

    # Remove old edges
    graph.remove_edge(u1, v1, k1)
    graph.remove_edge(u2, v2, k2)

    # Connect old edges to reconnection node
    graph.add_edge(other1, str(new_node), label=label1)
    graph.add_edge(other2, str(new_node), label=label2)

    # Add new open edge with specified label
    external_node = new_node + 1
    graph.add_node(str(external_node))
    graph.add_edge(str(new_node), str(external_node), label=new_label)

    return graph, label1, label2


# Computes ∏ Δ(j) for each j in labels.  Δ(j) = (-1)^(2j) × (2j+1).
# Keeps sign and magnitude separate to avoid overflow, then recombines.
def compute_delta_product(labels):
    """Compute product of Δ symbols."""
    evaluator = SpinNetworkEvaluator()
    total_sign_exponent = 0.0
    total_magnitude = 1.0
    for j in labels:
        sign_exp, mag = evaluator.delta_symbol(j, power=1)
        total_sign_exponent += sign_exp
        total_magnitude *= mag
    evaluator.cleanup()
    # Combine sign and magnitude
    sign = (-1.0) ** int(round(total_sign_exponent))
    return sign * total_magnitude


# Computes ∏ Θ(j1,j2,j3) for each triplet.
# Θ(j1,j2,j3) = (-1)^(j1+j2+j3) × (j1+j2+j3+1)! / [(j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)!].
def compute_theta_product(triplets):
    evaluator = SpinNetworkEvaluator()
    total_sign_exponent = 0.0
    total_magnitude = 1.0
    for j1, j2, j3 in triplets:
        sign_exp, mag = evaluator.theta_symbol(j1, j2, j3, power=1)
        total_sign_exponent += sign_exp
        total_magnitude *= mag
    evaluator.cleanup()
    # Combine sign and magnitude
    sign = (-1.0) ** int(round(total_sign_exponent))
    return sign * total_magnitude


# Main entry point: given original.graphml and two edge specs (e.g. "1-2" "3-4"),
# enumerates all admissible c values, computes p(c) = |Δ(c)/Θ(a,b,c) × ||G2||/||G1|||
# for each, prints a table, and verifies sum = 1.  Saves results to JSON.
def main():
    parser = argparse.ArgumentParser(
        description="Compute probabilities for all possible reconnection values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compute_all_probabilities.py drawn_graph.graphml 1-2 3-4

This will:
1. Compute original graph norm ||G₁||
2. For each possible new edge value satisfying triangle inequality:
   - Create reconnected graph
   - Compute reconnected norm ||G₂||
   - Compute probability p(c)
3. Verify that sum of all probabilities = 1
"""
    )
    parser.add_argument("original_file", help="Original graph file (.graphml)")
    parser.add_argument("edge1", help="First edge to reconnect (format: node1-node2)")
    parser.add_argument("edge2", help="Second edge to reconnect (format: node1-node2)")
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress intermediate output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.original_file):
        print(f"Error: Original file '{args.original_file}' not found.")
        sys.exit(1)

    print("\n" + "="*70)
    print("COMPUTE ALL RECONNECTION PROBABILITIES")
    print("="*70)

    # Load original graph
    print(f"\n[STEP 1] Loading original graph: {args.original_file}")
    original_graph = load_graph(args.original_file)
    print(f"  Nodes: {original_graph.number_of_nodes()}")
    print(f"  Edges: {original_graph.number_of_edges()}")

    # Get edge labels
    print(f"\n[STEP 2] Analyzing edges to reconnect...")
    print(f"  Edge 1: {args.edge1}")
    print(f"  Edge 2: {args.edge2}")

    # Create temporary graph to extract labels
    temp_graph, label1, label2 = reconnect_edges_with_label(
        original_graph, args.edge1, args.edge2, 0.0
    )

    print(f"  Edge 1 label: {label1}")
    print(f"  Edge 2 label: {label2}")

    # Calculate possible values
    possible_values = calculate_possible_values(label1, label2)
    print(f"\n[STEP 3] Possible new edge values: {possible_values}")
    print(f"  Number of possibilities: {len(possible_values)}")
    print(f"  (Based on triangle inequality: |{label1}-{label2}| ≤ c ≤ {label1}+{label2})")

    # Compute original norm
    print(f"\n[STEP 4] Computing original graph norm...")
    canon_G1, norm1 = _compute_norm_full(original_graph)
    print(f"  ||G₁|| = {norm1}")

    # Compute probability for each possible value
    print(f"\n[STEP 5] Computing probabilities for each possible value...")
    print("  " + "-"*66)

    results = []
    formula_entries = []

    for new_label in possible_values:
        print(f"\n  → New edge label: {new_label}")

        # Create reconnected graph
        reconnected_graph, _, _ = reconnect_edges_with_label(
            original_graph, args.edge1, args.edge2, new_label
        )

        # Compute reconnected norm directly from graph object (no temp files needed)
        if not args.quiet:
            print(f"    Computing ||G₂|| for c={new_label}...")
        canon_G2, norm2 = _compute_norm_full(reconnected_graph)
        print(f"    ||G₂|| = {norm2}")

        # Compute Delta
        delta = compute_delta_product([new_label])
        print(f"    Δ({new_label}) = {delta}")

        # Compute Theta
        theta = compute_theta_product([(label1, label2, new_label)])
        print(f"    Θ({label1}, {label2}, {new_label}) = {theta}")

        # Compute probability
        if theta == 0:
            print(f"    ⚠ Warning: Theta is zero!")
            probability = 0
        else:
            norm_ratio = norm2 / norm1 if norm1 != 0 else 0
            probability = abs((delta / theta) * norm_ratio)

        print(f"    ✓ p({new_label}) = {probability:.15e}")

        results.append({
            'new_label': float(new_label),
            'norm': float(norm2),
            'delta': float(delta),
            'theta': float(theta),
            'probability': float(probability)
        })

        norm_G2_formula = terms_to_formula_string(canon_G2)
        formula_entries.append(
            f"# c = {new_label}\n"
            f"abs(\n"
            f"    delta({new_label})\n"
            f"    / theta({label1}, {label2}, {new_label})\n"
            f"    * ({norm_G2_formula})\n"
            f"    / ({terms_to_formula_string(canon_G1)})\n"
            f")"
        )

    # Verify normalization
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\nOriginal graph norm: ||G₁|| = {norm1}")
    print(f"Edges reconnected: {args.edge1} (label={label1}) + {args.edge2} (label={label2})")
    print(f"\nProbability distribution:")
    print("  " + "-"*66)
    print(f"  {'New Edge':<12} {'||G₂||':<15} {'Δ':<10} {'Θ':<10} {'Probability':<15}")
    print("  " + "-"*66)

    total_probability = 0.0
    for r in results:
        print(f"  {r['new_label']:<12.1f} {r['norm']:<15.6e} {r['delta']:<10.3f} "
              f"{r['theta']:<10.3f} {r['probability']:<15.6e}")
        total_probability += r['probability']

    print("  " + "-"*66)
    print(f"  {'TOTAL':<12} {'':<15} {'':<10} {'':<10} {total_probability:<15.6e}")

    # Normalization test
    print(f"\n{'★'*70}")
    print("NORMALIZATION TEST")
    print(f"{'★'*70}")
    print(f"\nSum of all probabilities: {total_probability:.15e}")

    epsilon = 1e-10
    if abs(total_probability - 1.0) < epsilon:
        print(f"✓ PASSED: Probabilities sum to 1 (within ε={epsilon})")
        print(f"  Physical consistency verified!")
    else:
        print(f"✗ FAILED: Probabilities do not sum to 1")
        print(f"  Difference: {abs(total_probability - 1.0):.15e}")
        print(f"  This may indicate:")
        print(f"    - Numerical precision issues")
        print(f"    - Missing transitions")
        print(f"    - Incorrect formula implementation")

    # Save results
    results_file = args.original_file.replace('.graphml', '_all_probabilities.json')
    full_results = {
        'original_file': args.original_file,
        'original_norm': float(norm1),
        'edge1': args.edge1,
        'edge2': args.edge2,
        'edge1_label': float(label1),
        'edge2_label': float(label2),
        'possible_values': [float(v) for v in possible_values],
        'probabilities': results,
        'total_probability': float(total_probability),
        'normalization_test': abs(total_probability - 1.0) < epsilon
    }

    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")

    # Save symbolic formula .txt with one entry per possible c value
    formula_file = args.original_file.replace('.graphml', '_all_probability_formulas.txt')
    with open(formula_file, 'w') as f:
        f.write(f"# All reconnection probability formulas\n")
        f.write(f"# Edges: {args.edge1} (label={label1}) + {args.edge2} (label={label2})\n")
        f.write(f"# Evaluate each block with: python scripts/evaluate_formula.py \"<formula>\"\n\n")
        f.write("\n\n".join(formula_entries) + "\n")
    print(f"✓ Formulas saved to: {formula_file}")
    print("="*70 + "\n")

    return full_results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
