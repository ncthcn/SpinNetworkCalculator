#!/usr/bin/env python3
"""
Evaluate Spin Network Norm (Numerical)

This script evaluates the canonical spin network expression numerically using the
wigxjpf C++ library for fast Wigner 6j symbol computation.

Prerequisites:
    - Run 'python compute_norm.py' first to generate the canonical expression
    - wigxjpf must be installed (pip install pywigxjpf)

Usage:
    python evaluate_norm.py [input_file.graphml] [--max-j MAX_J]

Options:
    --max-j MAX_J    Maximum j value expected (default: auto-detect)

Output:
    Numerical value of the spin network norm
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import networkx as nx
from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.norm_reducer import canonicalise_terms, apply_kroneckers, expand_6j_symbolic
from src.spin_evaluator import evaluate_spin_network


def load_graph_from_file(file_path):
    """
    Load a graph from a GraphML file and ensure all nodes have valid positions.
    """
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


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate spin network norm numerically using wigxjpf backend",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default="drawn_graph_with_labels.graphml",
        help="Input GraphML file (default: drawn_graph_with_labels.graphml)"
    )
    parser.add_argument(
        "--max-j",
        type=int,
        default=None,
        help="Maximum j value for wigxjpf tables (default: auto-detect)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress intermediate output"
    )

    args = parser.parse_args()
    filepath = args.input_file

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        print(f"\nUsage: python {sys.argv[0]} [input_file.graphml] [--max-j MAX_J]")
        sys.exit(1)

    if not args.quiet:
        print("="*70)
        print("SPIN NETWORK NORM EVALUATION (Numerical)")
        print("="*70)
        print(f"Input file: {filepath}\n")

    # Load graph and perform reduction
    if not args.quiet:
        print("Loading and reducing graph...")

    graph = load_graph_from_file(filepath)
    glued_graph = glue_open_edges(graph)
    terms = reduce_all_cycles(glued_graph)

    if not args.quiet:
        print("  ✓ Graph reduction complete")

    # Apply Kronecker reductions
    if not args.quiet:
        print("Applying Kronecker reductions...")

    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)

    # Expand 6j to W6j
    if not args.quiet:
        print("Expanding 6j symbols to W6j...")

    for term in clean_terms:
        new = []
        for c in term["coeffs"]:
            if isinstance(c, dict) and c.get("type") == "6j":
                expanded = expand_6j_symbolic(c)
                new.extend(expanded)
            else:
                new.append(c)
        term["coeffs"] = new

    # Canonicalize
    if not args.quiet:
        print("Canonicalizing expression...")

    canon_terms = canonicalise_terms(clean_terms)

    if not args.quiet:
        print("  ✓ Symbolic computation complete")

    # Determine max spin value
    if args.max_j is not None:
        max_spin = args.max_j
    else:
        max_spin = 0
        for term in canon_terms:
            for coeff in term.get("coeffs", []):
                if isinstance(coeff, dict):
                    # Check args
                    for arg in coeff.get("args", ()):
                        if isinstance(arg, (int, float)):
                            max_spin = max(max_spin, int(arg))
                    # Check fixed values
                    for val in coeff.get("fixed", {}).values():
                        if isinstance(val, (int, float)):
                            max_spin = max(max_spin, int(val))
                    # Check range
                    range_info = coeff.get("range2", {})
                    if range_info:
                        max_spin = max(max_spin, range_info.get("Fmax", 0) // 2)

    max_two_j = max(100, max_spin * 2 + 10)  # Add buffer

    if not args.quiet:
        print(f"\nDetected max spin value: j = {max_spin}")
        print(f"Using max_two_j = {max_two_j} for wigxjpf tables")
        print("\n" + "="*70)
        print("NUMERICAL EVALUATION")
        print("="*70)

    # Evaluate numerically
    numerical_result = evaluate_spin_network(canon_terms, max_two_j=max_two_j)

    # Print result
    print("\n" + "="*70)
    print(f"✨ SPIN NETWORK NORM = {numerical_result:.15e}")
    print("="*70)

    if abs(numerical_result) < 1e-10:
        print("\n⚠️  Warning: Result is very close to zero!")
        print("This might indicate that some triangle conditions are not satisfied")
        print("or that the spin network configuration is forbidden by SU(2) recoupling rules.")


if __name__ == "__main__":
    main()
