#!/usr/bin/env python3
"""
Command-Line Graph Comparison Workflow (No GUI)

This is an alternative to compare_graphs.py that works without GUI.
Useful for:
- Systems with Tkinter compatibility issues
- Headless servers
- Scripted/batch workflows

Usage:
    python compare_graphs_cli.py original.graphml modified.graphml \\
        --flagged-edge 2 5 --flagged-vertex 5 \\
        --other-labels 1.0 1.0
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
from src.spin_evaluator import evaluate_spin_network


def load_graph_from_file(file_path):
    """Load a graph from a GraphML file."""
    graph = nx.read_graphml(file_path, force_multigraph=True)

    # Convert edge labels
    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except ValueError:
            pass

    # Ensure positions
    pos = nx.kamada_kawai_layout(graph)
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            graph.nodes[node]["pos"] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        else:
            graph.nodes[node]["pos"] = pos[node]

    return graph


def compute_norm(graph_file, quiet=True):
    """
    Compute the numerical norm of a spin network graph.
    Returns the norm value as a float.
    """
    if not quiet:
        print(f"\n{'='*70}")
        print(f"COMPUTING NORM: {graph_file}")
        print(f"{'='*70}")

    # Load graph
    graph = load_graph_from_file(graph_file)

    # Glue open edges
    if not quiet:
        print("Gluing open edges...")
    glued_graph = glue_open_edges(graph)

    # Reduce graph
    if not quiet:
        print("Reducing graph...")
    terms = reduce_all_cycles(glued_graph, animator=None)

    # Apply Kronecker reductions
    if not quiet:
        print("Applying Kronecker reductions...")
    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)

    # Expand 6j to W6j
    if not quiet:
        print("Expanding 6j symbols...")
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
    if not quiet:
        print("Canonicalizing...")
    canon_terms = canonicalise_terms(clean_terms)

    # Evaluate
    if not quiet:
        print("Evaluating numerically...")
    result = evaluate_spin_network(canon_terms)

    if not quiet:
        print(f"\n  → Norm = {result}")
        print(f"{'='*70}\n")

    return result


def compute_theta_product(labels):
    """
    Compute the product of theta symbols for given edge labels.
    Theta(j1, j2, j3) = (-1)^(j1+j2+j3) * (j1+j2+j3+1)! / [(j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)!]
    """
    from src.spin_evaluator import SpinNetworkEvaluator

    if len(labels) != 3:
        raise ValueError("Theta requires exactly 3 edge labels")

    evaluator = SpinNetworkEvaluator()
    j1, j2, j3 = labels
    result = evaluator.theta_symbol(j1, j2, j3, power=1)
    evaluator.cleanup()
    return result


def compute_delta_product(labels):
    """
    Compute the product of delta symbols for given edge labels.
    Delta(j) = (-1)^(2j) * (2j+1)
    """
    from src.spin_evaluator import SpinNetworkEvaluator

    evaluator = SpinNetworkEvaluator()
    result = 1.0
    for j in labels:
        result *= evaluator.delta_symbol(j, power=1)
    evaluator.cleanup()
    return result


def get_edge_label(graph, node1, node2, key=None):
    """Get label of edge between two nodes."""
    if key is not None:
        return graph.edges[node1, node2, key].get('label', '?')
    else:
        # Get first edge
        for k in graph[node1][node2]:
            return graph.edges[node1, node2, k].get('label', '?')
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare two spin network graphs (command-line, no GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison (no flagged edge data)
  python compare_graphs_cli.py original.graphml modified.graphml

  # With flagged edge information
  python compare_graphs_cli.py original.graphml modified.graphml \\
      --flagged-edge 2 5 --flagged-label 1.0 --flagged-vertex 5 \\
      --other-labels 1.0 1.0
"""
    )
    parser.add_argument("original_file", help="Original .graphml file")
    parser.add_argument("modified_file", help="Modified .graphml file")
    parser.add_argument(
        "--flagged-edge",
        nargs=2,
        type=int,
        metavar=("NODE1", "NODE2"),
        help="Flagged edge nodes (e.g., --flagged-edge 2 5)"
    )
    parser.add_argument(
        "--flagged-label",
        type=float,
        help="Label of the flagged edge (e.g., --flagged-label 1.0)"
    )
    parser.add_argument(
        "--flagged-vertex",
        type=int,
        help="Vertex ID where the flagged edge connects (e.g., --flagged-vertex 5)"
    )
    parser.add_argument(
        "--other-labels",
        nargs=2,
        type=float,
        metavar=("LABEL1", "LABEL2"),
        help="Labels of the other two edges at the flagged vertex"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="comparison_results.json",
        help="Output JSON file (default: comparison_results.json)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress intermediate output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.original_file):
        print(f"Error: Original file '{args.original_file}' not found.")
        sys.exit(1)

    if not os.path.exists(args.modified_file):
        print(f"Error: Modified file '{args.modified_file}' not found.")
        sys.exit(1)

    print("\n" + "="*70)
    print("SPIN NETWORK GRAPH COMPARISON (CLI)")
    print("="*70)

    # Compute original norm
    print("\n[STEP 1] Computing original graph norm...")
    original_norm = compute_norm(args.original_file, quiet=args.quiet)
    print(f"  → Original norm: {original_norm}")

    # Compute modified norm
    print("\n[STEP 2] Computing modified graph norm...")
    modified_norm = compute_norm(args.modified_file, quiet=args.quiet)
    print(f"  → Modified norm: {modified_norm}")

    # Compute ratio
    norm_ratio = modified_norm / original_norm if original_norm != 0 else None
    print(f"\n[STEP 3] Norm ratio: {norm_ratio}")

    # Process flagged edge data if provided
    flagged_data = None
    theta_val = None
    delta_val = None

    if args.flagged_edge and args.flagged_label and args.flagged_vertex and args.other_labels:
        print("\n[STEP 4] Processing flagged edge data...")

        n1, n2 = args.flagged_edge
        flagged_data = {
            'edge_nodes': (n1, n2),
            'edge_label': args.flagged_label,
            'vertex_id': args.flagged_vertex,
            'other_edge_labels': list(args.other_labels)
        }

        print(f"  Flagged edge: {n1} -- [{args.flagged_label}] -- {n2}")
        print(f"  Vertex: {args.flagged_vertex}")
        print(f"  Other labels: {args.other_labels}")

        # Compute theta and delta
        print("\n[STEP 5] Computing coefficients...")
        try:
            all_labels = [args.flagged_label] + list(args.other_labels)
            theta_val = compute_theta_product(all_labels)
            print(f"  Theta({all_labels[0]}, {all_labels[1]}, {all_labels[2]}) = {theta_val}")

            delta_val = compute_delta_product([args.flagged_label])
            print(f"  Delta({args.flagged_label}) = {delta_val}")
        except Exception as e:
            print(f"  ⚠ Error computing coefficients: {e}")
    else:
        print("\n[STEP 4] No flagged edge data provided (use --flagged-edge, --flagged-label, etc.)")

    # Display results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\nOriginal Graph: {args.original_file}")
    print(f"  Norm = {original_norm}")
    print(f"\nModified Graph: {args.modified_file}")
    print(f"  Norm = {modified_norm}")
    print(f"\nNorm Ratio: {norm_ratio}")

    if flagged_data:
        print(f"\nFlagged Edge:")
        print(f"  Nodes: {flagged_data['edge_nodes']}")
        print(f"  Label: {flagged_data['edge_label']}")
        print(f"  Vertex: {flagged_data['vertex_id']}")
        print(f"  Other labels: {flagged_data['other_edge_labels']}")

        if theta_val is not None:
            print(f"\nCoefficients:")
            print(f"  Theta = {theta_val}")
            print(f"  Delta = {delta_val}")

    # Save results
    results = {
        'original_file': args.original_file,
        'modified_file': args.modified_file,
        'original_norm': float(original_norm),
        'modified_norm': float(modified_norm),
        'norm_ratio': float(norm_ratio) if norm_ratio is not None else None,
        'flagged_edge': flagged_data,
        'theta': float(theta_val) if theta_val is not None else None,
        'delta': float(delta_val) if delta_val is not None else None
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
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
