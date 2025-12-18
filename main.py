import os
import networkx as nx
from drawing import draw_graph, compute_layout
from utils import vertex_satisfies_triangular_conditions
from reducer import reduce_all_cycles 
from gluer import glue_open_edges
import imageio


    # Function to check the triangular condition for all nodes
def check_triangular_condition(graph):
    """
    Check the triangular condition for all nodes in the graph.
    Skip nodes with non-numeric labels.
    """
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 3:  # Only check for nodes with exactly 3 edges
            labels = []
            for neighbor in neighbors:
                edge_data = graph[node][neighbor]
                for key, data in edge_data.items():
                    label = data.get("label", None)
                    if isinstance(label, (int, float)):  # Only consider numeric labels
                        labels.append(label)
            if len(labels) == 3 and not vertex_satisfies_triangular_conditions(labels):
                raise ValueError(f"Triangular condition not satisfied at node {node}")

# ------------------------
# Preparing 6j evaluation batch for C++ backend
# ------------------------

def collect_6j_requests(terms):
    """
    From the term expansion [(graph, coeff_list)], extract unique 6j-argument tuples to compute.
    Returns:
    requests: list of unique 6j arg tuples in doubled int form
    mapping: maps each term index -> list of (request_index, multiplicative factor index)
    """
    requests = []
    req_index = {}
    term_map = []

    for t_idx, (G, coeffs) in enumerate(terms):
        this_term_map = []
        for c in coeffs:
            if c["type"] == "6j":
                args = tuple(c["args2"])
                if args not in req_index:
                    req_index[args] = len(requests)
                    requests.append(args)
                this_term_map.append(req_index[args])
        term_map.append(this_term_map)

    return requests, term_map


def draw_curved_edge(x1, y1, x2, y2, curvature):
    """
    Draw a single curved edge between two points using a Bézier curve.
    """
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    dx = y2 - y1
    dy = x1 - x2
    control_x = mid_x + curvature * dx
    control_y = mid_y + curvature * dy

    # Draw the Bézier curve
    bezier_x = [x1, control_x, x2]
    bezier_y = [y1, control_y, y2]
    plt.plot(bezier_x, bezier_y, color="black")


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

def debug_graph_summary(G):
    print("Node degrees:")
    for v in G.nodes():
        print(f"  {v}: degree={G.degree(v)}")
    print("Parallel edges and labels:")
    for u, v in nx.Graph(G).edges():
        labels = [d.get("label") for k, d in G[u][v].items()]
        print(f"  {u}-{v}: labels={labels}, count={len(labels)}")
    cycles = nx.cycle_basis(nx.Graph(G))
    print(f"Cycle basis ({len(cycles)}): lengths={[len(C) for C in cycles]}")

# -------------------------------
# --- Main
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

    for t_idx, term in enumerate(terms):
        coeffs = term.get("coeffs", [])

        # Build formatted factor strings
        factors = []
        for c in coeffs:
            typ = c.get("type")
            fixed = c.get("fixed", {})

            if typ == "6j":
                # canonical order: [a,b,f,d,c,e]
                a = fixed.get("a", fixed.get("A", fixed.get("a")))
                b = fixed.get("b", fixed.get("B", fixed.get("b")))
                d = fixed.get("d", fixed.get("D", fixed.get("d")))
                cc = fixed.get("c", fixed.get("C", fixed.get("c")))
                e = fixed.get("e", fixed.get("E", fixed.get("e")))

                # sum range if present (doubled form)
                rng = c.get("sum_range2")
                if rng:
                    rng_str = f"[F={rng['Fmin']}...{rng['Fmax']} step 2, parity={rng['parity']}]"
                    factors.append(f"Σ_f{rng_str} 6j({a},{b},f,{d},{cc},{e})")
                else:
                    factors.append(f"6j({a},{b},f,{d},{cc},{e})")

            elif typ == "theta":
                a = fixed.get("a", fixed.get("A", fixed.get("a")))
                b = fixed.get("b", fixed.get("B", fixed.get("b")))
                cval = fixed.get("c", fixed.get("C", fixed.get("c")))
                factors.append(f"θ({a},{b},{cval})")

            elif typ == "delta":
                c1 = fixed.get("c", fixed.get("C", fixed.get("c")))
                d1 = fixed.get("d", fixed.get("D", fixed.get("d")))
                factors.append(f"δ({c1},{d1})")

            else:
                # Fallback: pretty print unknown coeff dict
                factors.append(str(c))

    product_str = " * ".join(factors) if factors else "1"

    print("\nNorm of the spin network:\n")
    print("|", product_str, "|")

    print("\nFinal reduced graph:")
    print(term["graph"])
    print()  # blank line between terms