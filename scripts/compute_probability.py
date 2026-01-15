#!/usr/bin/env python3
"""
Compute Reconnection Probability

Computes the probability for edge reconnection transitions using the formula:

p(c,...,z) = abs(Δ(c)⋅⋅⋅Δ(z) / [Θ(a,b,c)⋅⋅⋅Θ(x,y,z)] × ||G₂||/||G₁||)

Where:
- c is the new open edge created by reconnecting open edges a and b
- z is the new open edge created by reconnecting open edges x and y
- ||G₁|| is the norm of the original graph
- ||G₂|| is the norm of the reconnected graph

Usage:
    python compute_probability.py original.graphml reconnected.graphml
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


def load_graph(file_path):
    """Load graph from GraphML."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except ValueError:
            pass

    pos = nx.kamada_kawai_layout(graph)
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            graph.nodes[node]["pos"] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        else:
            graph.nodes[node]["pos"] = pos[node]

    return graph


def compute_norm(graph_file, quiet=True):
    """Compute spin network norm."""
    if not quiet:
        print(f"\nComputing norm for {graph_file}...")

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


def load_reconnection_data(recon_file):
    """Load reconnection data from JSON."""
    with open(recon_file, 'r') as f:
        return json.load(f)


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
        description="Compute reconnection probability for spin networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Formula:
  p(c,...,z) = abs(Δ(c)⋅⋅⋅Δ(z) / [Θ(a,b,c)⋅⋅⋅Θ(x,y,z)] × ||G₂||/||G₁||)

Example:
  python compute_probability.py original.graphml reconnected.graphml
"""
    )
    parser.add_argument("original_file", help="Original graph file (.graphml)")
    parser.add_argument("reconnected_file", help="Reconnected graph file (.graphml)")
    parser.add_argument(
        "--reconnection-data",
        help="Reconnection data file (.json). Auto-detected if not provided."
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress intermediate output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.original_file):
        print(f"Error: Original file '{args.original_file}' not found.")
        sys.exit(1)

    if not os.path.exists(args.reconnected_file):
        print(f"Error: Reconnected file '{args.reconnected_file}' not found.")
        sys.exit(1)

    # Auto-detect reconnection data file
    if not args.reconnection_data:
        args.reconnection_data = args.reconnected_file.replace('.graphml', '_reconnections.json')

    if not os.path.exists(args.reconnection_data):
        print(f"Error: Reconnection data file '{args.reconnection_data}' not found.")
        print(f"\nPlease specify it with --reconnection-data or ensure it's saved alongside the reconnected graph.")
        sys.exit(1)

    print("\n" + "="*70)
    print("SPIN NETWORK RECONNECTION PROBABILITY")
    print("="*70)

    # Step 1: Compute norms
    print("\n[STEP 1] Computing graph norms...")
    norm1 = compute_norm(args.original_file, quiet=args.quiet)
    print(f"  ||G₁|| (original) = {norm1}")

    norm2 = compute_norm(args.reconnected_file, quiet=args.quiet)
    print(f"  ||G₂|| (reconnected) = {norm2}")

    norm_ratio = norm2 / norm1 if norm1 != 0 else 0
    print(f"  ||G₂||/||G₁|| = {norm_ratio}")

    # Step 2: Load reconnection data
    print("\n[STEP 2] Loading reconnection data...")
    reconnections = load_reconnection_data(args.reconnection_data)
    print(f"  Number of reconnections: {len(reconnections)}")

    # Step 3: Extract labels
    print("\n[STEP 3] Extracting edge labels...")

    # Collect all new edge labels (c, ..., z)
    new_labels = []
    # Collect all theta triplets (a,b,c), ..., (x,y,z)
    theta_triplets = []

    for i, recon in enumerate(reconnections, 1):
        old_edge1 = recon['old_edges'][0]
        old_edge2 = recon['old_edges'][1]
        new_edge = recon['new_edge']

        label_a = old_edge1['label']
        label_b = old_edge2['label']
        label_c = new_edge['label']

        new_labels.append(label_c)
        theta_triplets.append((label_a, label_b, label_c))

        print(f"  Reconnection {i}:")
        print(f"    Old edges: {label_a}, {label_b}")
        print(f"    New edge: {label_c}")
        print(f"    Theta triplet: Θ({label_a}, {label_b}, {label_c})")

    # Step 4: Compute Delta product
    print("\n[STEP 4] Computing Δ product...")
    delta_product = compute_delta_product(new_labels)
    print(f"  Δ({', '.join(map(str, new_labels))}) = {delta_product}")

    # Step 5: Compute Theta product
    print("\n[STEP 5] Computing Θ product...")
    theta_product = compute_theta_product(theta_triplets)
    theta_str = ' × '.join([f"Θ({a},{b},{c})" for a, b, c in theta_triplets])
    print(f"  {theta_str} = {theta_product}")

    # Step 6: Compute probability
    print("\n[STEP 6] Computing probability...")

    if theta_product == 0:
        print("  ⚠ Warning: Theta product is zero!")
        probability = 0
    else:
        probability = abs((delta_product / theta_product) * norm_ratio)

    print(f"\n  p = abs(Δ / Θ × ||G₂||/||G₁||)")
    print(f"    = abs({delta_product} / {theta_product} × {norm_ratio})")
    print(f"    = abs({(delta_product / theta_product) * norm_ratio})")
    print(f"    = {probability}")

    # Display results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nOriginal graph norm: ||G₁|| = {norm1}")
    print(f"Reconnected graph norm: ||G₂|| = {norm2}")
    print(f"Norm ratio: ||G₂||/||G₁|| = {norm_ratio}")
    print(f"\nNumber of reconnections: {len(reconnections)}")
    print(f"New edge labels: {new_labels}")
    print(f"Δ product: {delta_product}")
    print(f"Θ product: {theta_product}")
    print(f"\n{'★'*70}")
    print(f"PROBABILITY: p = {probability:.15e}")
    print(f"{'★'*70}")

    # Save results
    results_file = args.reconnected_file.replace('.graphml', '_probability.json')
    results = {
        'original_file': args.original_file,
        'reconnected_file': args.reconnected_file,
        'original_norm': float(norm1),
        'reconnected_norm': float(norm2),
        'norm_ratio': float(norm_ratio),
        'num_reconnections': len(reconnections),
        'new_edge_labels': new_labels,
        'theta_triplets': theta_triplets,
        'delta_product': float(delta_product),
        'theta_product': float(theta_product),
        'probability': float(probability)
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("="*70 + "\n")

    return results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
