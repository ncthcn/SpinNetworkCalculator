import os
import networkx as nx
from drawing import draw_graph, compute_layout
from utils import check_triangular_condition
from graph_reducer import reduce_all_cycles 
from gluer import glue_open_edges
from LaTeX_rendering import save_latex_pdf
from norm_reducer import apply_kroneckers, expand_6j_symbolic, canonicalise_term

# Function to load a graph from a GraphML file
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
    pos = nx.kamada_kawai_layout(graph)  # Generate default positions
    for node in graph.nodes:
        if "x" in graph.nodes[node] and "y" in graph.nodes[node]:
            # If x and y attributes exist, use them to set the position
            graph.nodes[node]["pos"] = (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
        else:
            # Otherwise, use the default position
            graph.nodes[node]["pos"] = pos[node]
        
    return graph

def print_norm_expression(terms, LaTeX=False):
    """
    Print the norm expression in a readable format.
    """
    ORDER = {
        "6j": 0,
        "theta": 1,
        "delta": 2,
        "delta inverse": 3,
        "Kronecker": 4,
    }

    for term in terms:
        coeffs = term.get("coeffs", [])

        # sort coeff list by type priority
        coeffs = sorted(
            coeffs,
            key=lambda c: ORDER.get(c.get("type"), 999)
        )

        factors = []

        for c in coeffs:
            typ = c.get("type")
            fixed = c.get("fixed", {})

            if typ == "6j":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                d = fixed.get("d", fixed.get("D"))
                cc = fixed.get("c", fixed.get("C"))
                e = fixed.get("e", fixed.get("E"))

                rng = c.get("sum_range2")
                if rng:
                    f = c.get("sum_index", "f")
                    rng_str = (
                        f"[{f}={rng['Fmin']/2},\ldots, {rng['Fmax']/2} "
                        # f"step 1, parity={rng['parity']}]"
                    )
                    factors.append(
                        f"∑_{rng_str} 6j({a},{b},{f},{cc},{d},{e})"
                    )
                else:
                    f = fixed.get("f", fixed.get("F"))
                    factors.append(f"6j({a},{b},{f},{cc},{d},{e})")

            elif typ == "theta":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                cval = fixed.get("c", fixed.get("C"))
                factors.append(f"θ({a},{b},{cval})")

            elif typ == "delta":
                j = fixed.get("j", fixed.get("J"))
                factors.append(f"Δ({j})")

            elif typ == "delta inverse":
                j = fixed.get("j", fixed.get("J"))
                factors.append(f"Δ^{{-1}}({j})")

            elif typ == "Kronecker":
                c1 = fixed.get("c", fixed.get("C"))
                d1 = fixed.get("d", fixed.get("D"))
                factors.append(f"δ({c1},{d1})")

            else:
                factors.append(str(c))


    product_str = " * ".join(factors) if factors else "1"

    print("\nNorm of the spin network:\n")
    print("|", product_str, "|")

# -------------------------------
# ----------- Main --------------
# -------------------------------

if __name__ == "__main__":
    filepath = "drawn_graph_with_labels.graphml"
    if not os.path.exists(filepath):
        print(f"Error: '{filepath}' not found.")
        exit(1)

    # Load MultiGraph from GraphML
    # graph = nx.read_graphml(filepath, force_multigraph=True)
    graph = load_graph_from_file(filepath)

    # # Convert edge labels to float
    # for u, v, k, data in graph.edges(keys=True, data=True):
    #     data["label"] = float(data["label"])

    # Check triangular condition
    try:
        check_triangular_condition(graph)
        print("Triangular condition satisfied for all nodes.")
    except ValueError as e:
        print(e)
        exit(1)

    # Draw original graph
    print("Original Graph:")
    edge_labels_orig = {(u, v, k): d["label"] for u, v, k, d in graph.edges(keys=True, data=True)}
    draw_graph(graph, edge_labels_orig)

    # Glue with copy
    glued_graph = glue_open_edges(graph)

    # Compute layout for glued graph
    compute_layout(glued_graph, layout="auto")

    # Draw glued graph
    print("Glued Graph (theta graph):")
    edge_labels_glued = {(u, v, k): d["label"] for u, v, k, d in glued_graph.edges(keys=True, data=True)}
    draw_graph(glued_graph, edge_labels_glued)

    # Check planarity
    is_planar, embedding = nx.check_planarity(glued_graph)
    if is_planar:
        print("The glued graph is planar.")
    else:
        print("The glued graph is not planar.")

    # -------------------------------
    # Call the cycle reducer (includes 2-cycle reductions and F-moves)
    # -------------------------------

    terms = reduce_all_cycles(glued_graph)

    # -------------------------------
    # Print result: formal product string
    # -------------------------------   

    print_norm_expression(terms)

    save_latex_pdf(terms, filename="norm_expression.pdf")

    print("\nApplying Kronecker reductions...")
    clean_terms = []
    for T in terms:
        t = apply_kroneckers(T)
        if t is not None:
            clean_terms.append(t)
    
    for term in clean_terms:
        new = []
        for c in term["coeffs"]:
            if c["type"] == "6j":
                new.append(expand_6j_symbolic(c))
            else:
                new.append(c)
        term["coeffs"] = new
    
    canon_terms = [ canonicalise_term(T) for T in clean_terms ]
