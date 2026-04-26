#!/usr/bin/env python3
"""
Compute Reconnection Probability (Symbolic)

Produces a symbolic expression for the reconnection probability without
performing any numerical evaluation. The output text file can be fed directly
to evaluate_formula.py (or evaluate_formula.py --formula-file).

Formula:
    p = abs( [Δ(c)·Δ(z)·…] / [Θ(a,b,c)·Θ(x,y,z)·…]  ×  ||G₂|| / ||G₁|| )

where ||G₁||, ||G₂|| are the full symbolic norm expressions after graph
reduction (6j symbols, thetas, deltas, summation variables).

Usage:
    python scripts/compute_symbolic_probability.py original.graphml reconnected.graphml
    python scripts/compute_symbolic_probability.py original.graphml reconnected.graphml \\
        --reconnection-data reconnected_reconnections.json

Output:
    symbolic_probability.pdf  —  structural LaTeX rendering
    symbolic_probability.txt  —  full Python formula for evaluate_formula.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import textwrap

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.norm_reducer import canonicalise_terms, apply_kroneckers, expand_6j_symbolic
from src.LaTeX_rendering import terms_to_formula_string, save_formula_txt, latex_formatting, _sanitize_py


# -----------------------------------------------------------------------
# Graph loading (shared pattern with compute_norm.py)
# -----------------------------------------------------------------------

def load_graph_from_file(file_path):
    graph = nx.read_graphml(file_path, force_multigraph=True)
    for u, v, data in graph.edges(data=True):
        try:
            data["label"] = float(data["label"])
        except (ValueError, TypeError):
            pass
    pos = nx.kamada_kawai_layout(graph)
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            graph.nodes[node]["pos"] = (
                float(graph.nodes[node]["x"]),
                float(graph.nodes[node]["y"]),
            )
        else:
            graph.nodes[node]["pos"] = pos[node]
    return graph


# -----------------------------------------------------------------------
# Symbolic reduction pipeline (returns canonical terms, no numerical eval)
# -----------------------------------------------------------------------

def symbolic_reduction(graph):
    """Run the full symbolic pipeline and return canonical terms."""
    glued = glue_open_edges(graph)
    terms = reduce_all_cycles(glued, animator=None)

    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)

    for term in clean_terms:
        new = []
        for c in term["coeffs"]:
            if isinstance(c, dict) and c.get("type") == "6j":
                new.extend(expand_6j_symbolic(c))
            else:
                new.append(c)
        term["coeffs"] = new

    return canonicalise_terms(clean_terms)


# -----------------------------------------------------------------------
# PDF rendering for symbolic probability
# -----------------------------------------------------------------------

def _latex_norm(terms):
    """Render canonical terms as a flat LaTeX string (cdot-joined)."""
    all_factors = latex_formatting(terms)
    flat = [f for sublist in all_factors for f in sublist]
    return r" \cdot ".join(flat) if flat else r"1"


def save_symbolic_probability_pdf(
    canon_G1, canon_G2, delta_parts, theta_parts,
    reconnections, filename="symbolic_probability.pdf"
):
    """
    Render a two-panel PDF:
      Top:    structural formula  p = |Δ(c)/Θ(a,b,c) × N₂/N₁|
      Bottom: expanded norm expressions for N₁ and N₂
    """
    norm1_latex = _latex_norm(canon_G1)
    norm2_latex = _latex_norm(canon_G2)

    delta_latex = r" \cdot ".join(
        rf"\Delta_{{{c}}}" for c in delta_parts
    ) or "1"
    theta_latex = r" \cdot ".join(
        rf"\Theta\left({a},{b},{c}\right)" for a, b, c in theta_parts
    ) or "1"

    struct_expr = (
        rf"p = \left| \frac{{{delta_latex}}}{{{theta_latex}}}"
        rf"\cdot \frac{{\|G_2\|}}{{\|G_1\|}} \right|"
    )
    norm1_expr = rf"\|G_1\| = {norm1_latex}"
    norm2_expr = rf"\|G_2\| = {norm2_latex}"

    fig, axes = plt.subplots(3, 1, figsize=(12, 5))
    for ax in axes:
        ax.axis('off')

    axes[0].text(0.5, 0.5, f"${struct_expr}$",
                 ha='center', va='center', fontsize=14, wrap=True)
    axes[1].text(0.05, 0.5, f"${norm1_expr}$",
                 ha='left', va='center', fontsize=10, wrap=True)
    axes[2].text(0.05, 0.5, f"${norm2_expr}$",
                 ha='left', va='center', fontsize=10, wrap=True)

    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved symbolic probability PDF to {filename}")


# -----------------------------------------------------------------------
# Build the probability formula string for evaluate_formula.py
# -----------------------------------------------------------------------

def build_probability_formula(canon_G1, canon_G2, delta_labels, theta_triplets):
    """
    Assemble the full evaluatable probability formula string:
        abs( (delta(c)*…) / (theta(a,b,c)*…) * NORM_G2 / NORM_G1 )
    """
    norm_G1 = terms_to_formula_string(canon_G1)
    norm_G2 = terms_to_formula_string(canon_G2)

    delta_str = " * ".join(f"delta({_sanitize_py(str(c))})" for c in delta_labels) or "1"
    theta_str = " * ".join(
        f"theta({_sanitize_py(str(a))}, {_sanitize_py(str(b))}, {_sanitize_py(str(c))})"
        for a, b, c in theta_triplets
    ) or "1"

    return (
        f"abs(\n"
        f"    ({delta_str})\n"
        f"    / ({theta_str})\n"
        f"    * ({norm_G2})\n"
        f"    / ({norm_G1})\n"
        f")"
    )


def save_probability_formula_txt(formula, filename):
    base = filename.replace("\\", "/").split("/")[-1]
    with open(filename, "w") as f:
        f.write(f"# Symbolic reconnection probability — {base}\n")
        f.write(
            f"# Evaluate: python scripts/evaluate_formula.py "
            f"\"$(grep -v '^#' {base} | tr -d '\\n')\"\n"
        )
        f.write(formula + "\n")
    print(f"Saved probability formula to {filename}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute symbolic reconnection probability expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("original_file", help="Original graph (.graphml)")
    parser.add_argument("reconnected_file", help="Reconnected graph (.graphml)")
    parser.add_argument(
        "--reconnection-data",
        help="Reconnection JSON (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out-pdf", default="symbolic_probability.pdf",
        help="Output PDF filename (default: symbolic_probability.pdf)",
    )
    parser.add_argument(
        "--out-txt", default="symbolic_probability.txt",
        help="Output text formula filename (default: symbolic_probability.txt)",
    )

    args = parser.parse_args()

    for path in (args.original_file, args.reconnected_file):
        if not os.path.exists(path):
            print(f"Error: '{path}' not found.")
            sys.exit(1)

    if not args.reconnection_data:
        args.reconnection_data = args.reconnected_file.replace(
            ".graphml", "_reconnections.json"
        )
    if not os.path.exists(args.reconnection_data):
        print(f"Error: Reconnection data '{args.reconnection_data}' not found.")
        sys.exit(1)

    print("=" * 70)
    print("SPIN NETWORK SYMBOLIC PROBABILITY")
    print("=" * 70)

    # Step 1 — symbolic reduction of both graphs
    print("\n[1/4] Reducing G₁ symbolically...")
    G1 = load_graph_from_file(args.original_file)
    canon_G1 = symbolic_reduction(G1)
    print(f"      → {len(canon_G1)} canonical term(s)")

    print("[2/4] Reducing G₂ symbolically...")
    G2 = load_graph_from_file(args.reconnected_file)
    canon_G2 = symbolic_reduction(G2)
    print(f"      → {len(canon_G2)} canonical term(s)")

    # Step 2 — load reconnection data
    print("[3/4] Loading reconnection data...")
    with open(args.reconnection_data) as f:
        reconnections = json.load(f)

    delta_labels = []
    theta_triplets = []
    for recon in reconnections:
        a = recon["old_edges"][0]["label"]
        b = recon["old_edges"][1]["label"]
        c = recon["new_edge"]["label"]
        delta_labels.append(c)
        theta_triplets.append((a, b, c))
        print(f"      Reconnection: Θ({a},{b},{c}),  Δ({c})")

    # Step 3 — also save individual norm formula files
    print("[4/4] Saving outputs...")
    base_orig = os.path.splitext(os.path.basename(args.original_file))[0]
    base_recon = os.path.splitext(os.path.basename(args.reconnected_file))[0]
    save_formula_txt(canon_G1, f"{base_orig}_norm.txt")
    save_formula_txt(canon_G2, f"{base_recon}_norm.txt")

    # Step 4 — build and save probability formula
    prob_formula = build_probability_formula(
        canon_G1, canon_G2, delta_labels, theta_triplets
    )
    save_probability_formula_txt(prob_formula, args.out_txt)
    save_symbolic_probability_pdf(
        canon_G1, canon_G2, delta_labels, theta_triplets,
        reconnections, filename=args.out_pdf,
    )

    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)
    print(f"  {args.out_pdf}            — structural LaTeX PDF")
    print(f"  {args.out_txt}            — full Python formula (evaluate_formula.py input)")
    print(f"  {base_orig}_norm.txt      — norm formula for G₁")
    print(f"  {base_recon}_norm.txt     — norm formula for G₂")
    print()
    print("Evaluate numerically:")
    print(
        f'  python scripts/evaluate_formula.py '
        f'"$(grep -v \'^#\' {args.out_txt} | tr -d \'\\n\')"'
    )
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"\nError: {e}")
        traceback.print_exc()
        sys.exit(1)
