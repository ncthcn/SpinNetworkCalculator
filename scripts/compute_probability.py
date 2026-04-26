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
from src.LaTeX_rendering import terms_to_formula_string, save_formula_txt, _sanitize_py


# Standard graph loader shared by all scripts: reads GraphML, coerces labels
# to float, populates "pos" from stored x/y or kamada-kawai fallback.
def load_graph_from_file(file_path):
    """Load a graph from a GraphML file and ensure all nodes have valid positions."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    # Convert edge labels to float or keep as string
    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except ValueError:
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
        print(f"\nComputing norm for {graph_file}...")

    graph = load_graph_from_file(graph_file)
    return compute_norm_from_graph(graph, quiet=quiet)


# Core norm pipeline: glue → reduce → Kronecker → expand 6j → canonicalise
# → evaluate. Returns a float.
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


# Reads the reconnection JSON produced by transition_to.py, which records
# old_edges (label_a, label_b) and new_edge (label_c) for each reconnection.
def load_reconnection_data(recon_file):
    """Load reconnection data from JSON."""
    with open(recon_file, 'r') as f:
        return json.load(f)


# Computes ∏ Δ(j) for each j in labels.  Δ(j) = (-1)^(2j) × (2j+1).
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


# Computes p = |Δ(c)/Θ(a,b,c) × ||G2||/||G1||| for a specific already-saved
# reconnection. Reads reconnection labels from a _reconnections.json file
# (produced by transition_to.py), then evaluates both norms numerically.
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
    G1 = load_graph_from_file(args.original_file)
    canon_G1, norm1 = _compute_norm_full(G1)
    print(f"  ||G₁|| (original) = {norm1}")

    G2 = load_graph_from_file(args.reconnected_file)
    canon_G2, norm2 = _compute_norm_full(G2)
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

    # Save symbolic formula .txt
    delta_str = " * ".join(f"delta({_sanitize_py(str(c))})" for c in new_labels) or "1"
    theta_str = " * ".join(
        f"theta({_sanitize_py(str(a))}, {_sanitize_py(str(b))}, {_sanitize_py(str(c))})"
        for a, b, c in theta_triplets
    ) or "1"
    norm_G1_formula = terms_to_formula_string(canon_G1)
    norm_G2_formula = terms_to_formula_string(canon_G2)
    prob_formula = (
        f"abs(\n"
        f"    ({delta_str})\n"
        f"    / ({theta_str})\n"
        f"    * ({norm_G2_formula})\n"
        f"    / ({norm_G1_formula})\n"
        f")"
    )
    formula_file = args.reconnected_file.replace('.graphml', '_probability_formula.txt')
    with open(formula_file, 'w') as f:
        f.write(f"# Reconnection probability formula\n")
        f.write(f"# Evaluate: python scripts/evaluate_formula.py --formula-file {formula_file}\n")
        f.write(prob_formula + "\n")
    print(f"✓ Formula saved to: {formula_file}")
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
