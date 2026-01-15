#!/usr/bin/env python3
"""
Graph Comparison Workflow

Automates the process of:
1. Loading an initial graph
2. Computing its norm
3. Opening GUI to modify and flag edges
4. Computing the new norm
5. Returning comparison data (old norm, new norm, flagged edge info)

This is used to compute transition probabilities between spin network states.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import subprocess
import json
import tkinter as tk
from modify_graph import GraphModifier
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
    print(f"\n{'='*70}")
    print(f"COMPUTING NORM: {graph_file}")
    print(f"{'='*70}")

    # Load graph
    if not quiet:
        print("Loading graph...")
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

    print(f"\n  → Norm = {result}")
    print(f"{'='*70}\n")

    return result


def extract_flagged_info(flagged_file):
    """Extract flagged edge information from the text file."""
    if not os.path.exists(flagged_file):
        return None

    data = {}
    with open(flagged_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("Edge nodes:"):
            # Parse tuple string
            nodes_str = line.split(":", 1)[1].strip()
            data['edge_nodes'] = eval(nodes_str)
        elif line.startswith("Edge label:"):
            label_str = line.split(":", 1)[1].strip()
            try:
                data['edge_label'] = float(label_str)
            except:
                data['edge_label'] = label_str
        elif line.startswith("Vertex ID:"):
            data['vertex_id'] = int(line.split(":", 1)[1].strip())
        elif line.startswith("Other edge labels:"):
            labels_str = line.split(":", 1)[1].strip()
            data['other_edge_labels'] = eval(labels_str)

    return data


def compute_theta_product(labels):
    """
    Compute the product of theta symbols for given edge labels.
    Theta(j1, j2, j3) = (-1)^(j1+j2+j3) * (j1+j2+j3+1)! / [(j1+j2-j3)!(j1-j2+j3)!(-j1+j2+j3)!]
    """
    from src.spin_evaluator import SpinNetworkEvaluator

    if len(labels) != 3:
        raise ValueError("Theta requires exactly 3 edge labels")

    evaluator = SpinNetworkEvaluator()  # __init__ automatically initializes
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

    evaluator = SpinNetworkEvaluator()  # __init__ automatically initializes
    result = 1.0
    for j in labels:
        result *= evaluator.delta_symbol(j, power=1)
    evaluator.cleanup()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compare two spin network graphs and compute transition data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "input_file",
        help="Input .graphml file (original graph)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="modified_graph.graphml",
        help="Output filename for modified graph (default: modified_graph.graphml)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress intermediate output"
    )
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Skip GUI (use existing modified graph)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print("\n" + "="*70)
    print("SPIN NETWORK GRAPH COMPARISON WORKFLOW")
    print("="*70)

    # Step 1: Compute original norm
    print("\n[STEP 1] Computing original graph norm...")
    original_norm = compute_norm(args.input_file, quiet=args.quiet)

    # Step 2: Open GUI for modification (unless skipped)
    if not args.skip_gui:
        print("\n[STEP 2] Opening graph modifier GUI...")
        print("  → Modify the graph and flag an open edge")
        print("  → Press 'Save & Exit' when done\n")

        root = tk.Tk()
        modifier = GraphModifier(root, args.input_file)
        root.mainloop()

        # Check if file was saved
        if not os.path.exists(args.output):
            print(f"\nError: Modified graph not saved to '{args.output}'")
            print("Please save the graph before exiting.")
            sys.exit(1)
    else:
        print("\n[STEP 2] Skipping GUI (using existing modified graph)...")

    # Step 3: Compute new norm
    print("\n[STEP 3] Computing modified graph norm...")
    modified_norm = compute_norm(args.output, quiet=args.quiet)

    # Step 4: Extract flagged edge data
    print("\n[STEP 4] Extracting flagged edge information...")
    flagged_file = args.output.replace('.graphml', '_flagged.txt')
    flagged_data = extract_flagged_info(flagged_file)

    if flagged_data is None:
        print("  ⚠ Warning: No flagged edge data found")
        print(f"  Expected file: {flagged_file}")
        flagged_data = {}

    # Step 5: Compute theta and delta ratios
    print("\n[STEP 5] Computing theta and delta coefficients...")

    theta_ratio = None
    delta_ratio = None

    if 'other_edge_labels' in flagged_data and len(flagged_data['other_edge_labels']) >= 2:
        try:
            # Get all three labels at the vertex (flagged + 2 others)
            all_labels = [flagged_data['edge_label']] + flagged_data['other_edge_labels'][:2]

            # Compute theta
            theta_val = compute_theta_product(all_labels)
            print(f"  Theta({all_labels[0]}, {all_labels[1]}, {all_labels[2]}) = {theta_val}")

            # Compute delta for flagged edge
            delta_val = compute_delta_product([flagged_data['edge_label']])
            print(f"  Delta({flagged_data['edge_label']}) = {delta_val}")

            # For transition probability: ratio involves these coefficients
            theta_ratio = theta_val
            delta_ratio = delta_val

        except Exception as e:
            print(f"  ⚠ Error computing coefficients: {e}")

    # Step 6: Display results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\nOriginal Graph: {args.input_file}")
    print(f"  Norm = {original_norm}")
    print(f"\nModified Graph: {args.output}")
    print(f"  Norm = {modified_norm}")
    print(f"\nNorm Ratio: {modified_norm / original_norm if original_norm != 0 else 'undefined'}")

    if flagged_data:
        print(f"\nFlagged Edge:")
        print(f"  Nodes: {flagged_data.get('edge_nodes', 'N/A')}")
        print(f"  Label: {flagged_data.get('edge_label', 'N/A')}")
        print(f"  Vertex: {flagged_data.get('vertex_id', 'N/A')}")
        print(f"  Other labels at vertex: {flagged_data.get('other_edge_labels', 'N/A')}")

        if theta_ratio is not None:
            print(f"\nCoefficients:")
            print(f"  Theta = {theta_ratio}")
            print(f"  Delta = {delta_ratio}")

    # Save results to JSON
    results_file = args.output.replace('.graphml', '_comparison.json')
    results = {
        'original_file': args.input_file,
        'modified_file': args.output,
        'original_norm': float(original_norm),
        'modified_norm': float(modified_norm),
        'norm_ratio': float(modified_norm / original_norm) if original_norm != 0 else None,
        'flagged_edge': flagged_data,
        'theta': float(theta_ratio) if theta_ratio is not None else None,
        'delta': float(delta_ratio) if delta_ratio is not None else None
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {results_file}")
    print("="*70 + "\n")

    # Return results for programmatic use
    return results


if __name__ == "__main__":
    results = main()
