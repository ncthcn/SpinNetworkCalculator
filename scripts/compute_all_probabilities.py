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
import tempfile

from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.norm_reducer import canonicalise_terms, apply_kroneckers, expand_6j_symbolic
from src.spin_evaluator import evaluate_spin_network, SpinNetworkEvaluator


def load_graph(file_path):
    """Load graph from GraphML."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except (ValueError, KeyError):
            pass

    return graph


def compute_norm(graph_file, quiet=True):
    """Compute spin network norm."""
    if not quiet:
        print(f"Computing norm for {graph_file}...")

    graph = load_graph(graph_file)
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

    if not quiet:
        print(f"  → Norm = {result}")

    return result


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


def reconnect_edges_with_label(graph, edge1_spec, edge2_spec, new_label):
    """
    Reconnect two edges with a specific new label.
    Returns modified graph.
    """
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


def compute_delta_product(labels):
    """Compute product of Δ symbols."""
    evaluator = SpinNetworkEvaluator()
    result = 1.0
    for j in labels:
        result *= evaluator.delta_symbol(j, power=1)
    evaluator.cleanup()
    return result


def compute_theta_product(triplets):
    """
    Compute product of Θ symbols.
    triplets: list of (j1, j2, j3) tuples
    """
    evaluator = SpinNetworkEvaluator()
    result = 1.0
    for j1, j2, j3 in triplets:
        result *= evaluator.theta_symbol(j1, j2, j3, power=1)
    evaluator.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute probabilities for all possible reconnection values",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compute_all_probabilities.py drawn_graph_with_labels.graphml 1-2 3-4

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
    norm1 = compute_norm(args.original_file, quiet=args.quiet)
    print(f"  ||G₁|| = {norm1}")

    # Compute probability for each possible value
    print(f"\n[STEP 5] Computing probabilities for each possible value...")
    print("  " + "-"*66)

    results = []
    temp_files = []

    for new_label in possible_values:
        print(f"\n  → New edge label: {new_label}")

        # Create reconnected graph
        reconnected_graph, _, _ = reconnect_edges_with_label(
            original_graph, args.edge1, args.edge2, new_label
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.graphml', delete=False) as f:
            temp_file = f.name
            temp_files.append(temp_file)

        nx.write_graphml(reconnected_graph, temp_file)

        # Compute reconnected norm
        if not args.quiet:
            print(f"    Computing ||G₂|| for c={new_label}...")
        norm2 = compute_norm(temp_file, quiet=args.quiet)
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

    # Cleanup temp files
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except:
            pass

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
