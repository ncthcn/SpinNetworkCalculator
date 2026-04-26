#!/usr/bin/env python3
"""
Compute Spin Network Norm (Symbolic)

This script computes the symbolic norm expression of a spin network from a GraphML file.
It performs all graph reductions and produces LaTeX PDFs showing the canonical form.

Usage:
    python compute_norm.py [input_file.graphml] [--animate]

Options:
    --animate : Generate animated GIF and PDF slideshow of reduction process

Output:
    - norm_expression.pdf: Raw norm expression after graph reduction
    - canon_norm_expression.pdf: Canonicalized expression with simplified coefficients
    - reduction.gif (if --animate): Animated visualization of reduction steps
    - reduction_slides.pdf (if --animate): PDF slideshow of all steps

The script does NOT perform numerical evaluation - use evaluate_norm.py for that.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import networkx as nx

from src.drawing import compute_layout, draw_graph, plot_kuratowski
from src.gluer import glue_open_edges
from src.graph_reducer import reduce_all_cycles
from src.LaTeX_rendering import save_formula_txt, save_latex_pdf
from src.norm_reducer import apply_kroneckers, canonicalise_terms, expand_6j_symbolic
from src.reduction_animator import ReductionAnimator
from src.utils import check_triangular_condition


# Reads a .graphml file produced by graph.py, converts edge labels to float
# where possible, and populates the "pos" node attribute from stored x/y
# coordinates (or from kamada-kawai layout as a fallback).
# Returns a NetworkX MultiGraph ready for gluing and reduction.
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
    pos = nx.kamada_kawai_layout(graph)  # Generate default positions
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            # If x and y attributes exist, use them to set the position
            graph.nodes[node]["pos"] = (
                float(graph.nodes[node]["x"]),
                float(graph.nodes[node]["y"]),
            )
        else:
            # Otherwise, use the default position
            graph.nodes[node]["pos"] = pos[node]

    return graph


# Terminal pretty-printer for the coefficient list produced by reduce_all_cycles.
# Iterates over each term and each coefficient dict, printing a human-readable
# product string. Used only for quick inspection; the authoritative output is
# the LaTeX PDF produced by save_latex_pdf.
def print_norm_expression(terms):
    ORDER = {
        "sum": 0,
        "sign": 1,
        "W6j": 2,
        "6j": 3,
        "theta": 4,
        "delta": 5,
        "delta inverse": 5,
        "Kronecker": 6,
    }

    for term in terms:
        coeffs = term.get("coeffs", [])

        # sort coeff list by type priority
        coeffs = sorted(coeffs, key=lambda c: ORDER.get(c.get("type"), 999))

        factors = []

        for c in coeffs:
            typ = c.get("type")
            fixed = c.get("fixed", {})

            if typ == "sum":
                f = c.get("index", "f")
                rng = c.get("range2")
                if rng:
                    rng_str = f"[{f}={rng['Fmin'] / 2},..., {rng['Fmax'] / 2}]"
                    factors.append(f"∑_{rng_str}")
                else:
                    factors.append(f"∑_{{f}}")

            elif typ == "sign":
                args = fixed.get("args", [])
                arg_strs = []
                for c, val in args:
                    if c is not None:
                        arg_strs.append(f"{c}{val}")
                    else:
                        arg_strs.append(f"+{val}")
                factors.append("(-1)^{{.join(arg_strs)}}")

            elif typ == "W6j":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                d = fixed.get("d", fixed.get("D"))
                c = fixed.get("c", fixed.get("C"))
                e = fixed.get("e", fixed.get("E"))
                f = fixed.get("f", fixed.get("F"))
                factors.append(f"W6j({a},{b},{f},{c},{d},{e})")

            elif typ == "6j":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                d = fixed.get("d", fixed.get("D"))
                c = fixed.get("c", fixed.get("C"))
                e = fixed.get("e", fixed.get("E"))
                f = fixed.get("f", fixed.get("F"))
                factors.append(f"6j({a},{b},{f},{c},{d},{e})")

            elif typ == "theta":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                cval = fixed.get("c", fixed.get("C"))
                factors.append(f"θ({a},{b},{cval})")

            elif typ == "delta":
                j = fixed.get("j", fixed.get("J"))
                power = fixed.get("power", 1)
                if power != 1:
                    factors.append(f"Δ({j})^{power}")
                else:
                    factors.append(f"Δ({j})")

            elif typ == "Kronecker":
                c1 = fixed.get("c", fixed.get("C"))
                d1 = fixed.get("d", fixed.get("D"))
                factors.append(f"δ({c1},{d1})")

            else:
                factors.append(str(c))

    product_str = " * ".join(factors) if factors else "1"

    print("\nNorm of the spin network:\n")
    print("|", product_str, "|")


# Entry point for the symbolic computation workflow:
#   load → validate → glue → reduce → Kronecker → expand 6j → canonicalise → PDF
# With --animate, captures each reduction step as a PNG and compiles a GIF.
# Does NOT evaluate numerically; use evaluate_norm.py for that.
# Draws the Kuratowski subgraph (a K5 or K3,3 subdivision) returned by
def main():
    # Parse command line arguments
    filepath = "drawn_graph.graphml"
    animate = False
    strict_planarity = False

    for arg in sys.argv[1:]:
        if arg == "--animate":
            animate = True
        elif arg == "--strict-planarity":
            strict_planarity = True
        elif not arg.startswith("--"):
            filepath = arg

    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        print(
            f"\nUsage: python {sys.argv[0]} [input_file.graphml] [--animate] [--strict-planarity]"
        )
        sys.exit(1)

    print("=" * 70)
    print("SPIN NETWORK NORM COMPUTATION (Symbolic)")
    print("=" * 70)
    print(f"Input file: {filepath}")
    if animate:
        print("Animation: ENABLED 🎬")
    print()

    # Initialize animator if requested
    animator = None
    if animate:
        animator = ReductionAnimator(output_dir="reduction_steps")
        print("📹 Animation mode: Capturing reduction steps...")

    # Load graph
    print("Loading graph...")
    graph = load_graph_from_file(filepath)
    print(
        f"  Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges"
    )

    # Check triangular condition
    try:
        check_triangular_condition(graph)
        print("  ✓ Triangular condition satisfied for all nodes")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
        sys.exit(1)

    # Draw original graph
    print("\nDrawing original graph...")
    edge_labels_orig = {
        (u, v, k): d["label"] for u, v, k, d in graph.edges(keys=True, data=True)
    }
    draw_graph(graph, edge_labels_orig)

    # Capture original graph
    if animator:
        animator.add_step(
            graph,
            title="Original Spin Network",
            description=f"Input graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges before any transformations.",
            operation="initial",
        )

    # Glue with copy
    print("\nGluing open edges (creating theta graph)...")
    glued_graph = glue_open_edges(graph)

    # Compute layout for glued graph
    compute_layout(glued_graph, layout="auto")

    # Draw glued graph
    print("Drawing glued graph...")
    edge_labels_glued = {
        (u, v, k): d["label"] for u, v, k, d in glued_graph.edges(keys=True, data=True)
    }
    draw_graph(glued_graph, edge_labels_glued)

    # Capture glued graph
    if animator:
        animator.add_step(
            glued_graph,
            title="After Gluing Open Edges",
            description=f"Open edges glued to form theta graph. Now has {glued_graph.number_of_nodes()} nodes and {glued_graph.number_of_edges()} edges.",
            operation="glue",
        )

    # Check planarity — counterexample=True returns the Kuratowski subgraph on failure
    is_planar, certificate = nx.check_planarity(glued_graph, counterexample=True)
    if is_planar:
        print("  ✓ The glued graph is planar")
    else:
        out_png = os.path.splitext(filepath)[0] + "_kuratowski.png"
        plot_kuratowski(certificate, out_png)
        if strict_planarity:
            print("  ✗ Non-planar graph — aborting (--strict-planarity is set)")
            sys.exit(1)
        print(
            "  ⚠ Non-planar graph — proceeding with cycle-basis fallback"
            " (pass --strict-planarity to abort instead)"
        )

    # Perform graph reduction
    print("\nPerforming graph reduction (F-moves, triangle reductions)...")
    if animate:
        print("  (Capturing intermediate steps for animation...)")
    terms = reduce_all_cycles(glued_graph, animator=animator)

    # Print symbolic result
    print_norm_expression(terms)

    # Apply Kronecker reductions
    print("\nApplying Kronecker reductions...")
    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)

    # Expand 6j to W6j
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
    print("Canonicalizing expression...")
    canon_terms = canonicalise_terms(clean_terms)

    # Save canonical LaTeX and plain-text formula
    print("\nSaving canonical norm expression...")
    save_latex_pdf(canon_terms, filename="canon_norm_expression.pdf")
    print("  ✓ Saved: canon_norm_expression.pdf")
    save_formula_txt(canon_terms, "canon_norm_expression.txt")
    print("  ✓ Saved: canon_norm_expression.txt")

    # Generate animations if requested
    if animator:
        print("\n" + "=" * 70)
        print("GENERATING ANIMATIONS")
        print("=" * 70)

        # Show summary
        animator.summary()

        # Save GIF
        print("\nGenerating animated GIF...")
        animator.save_gif("reduction.gif", duration=2.0)

        # Save PDF slideshow
        print("\nGenerating PDF slideshow...")
        animator.save_slides_pdf("reduction_slides.pdf")

        print("\n" + "=" * 70)
        print("ANIMATION FILES CREATED")
        print("=" * 70)
        print("  📊 Individual frames: reduction_steps/step_*.png")
        print("  🎬 Animated GIF: reduction.gif")
        print("  📄 PDF Slideshow: reduction_slides.pdf")

    print("\n" + "=" * 70)
    print("SYMBOLIC COMPUTATION COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  • python scripts/evaluate_norm.py            (numerical evaluation)")
    print(
        "  • python scripts/evaluate_formula.py \"$(grep -v '^#' canon_norm_expression.txt)\"  (formula evaluator)"
    )
    if animate:
        print("\nTo view the animation:")
        print("  - Open reduction.gif in any image viewer")
        print("  - Open reduction_slides.pdf to see all steps as slides")


if __name__ == "__main__":
    main()
