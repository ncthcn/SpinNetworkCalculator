import networkx as nx
import copy
# -------------------------------
# --- Glue graphs along open ends
# -------------------------------

def glue_open_edges(graph, offset=10):
    G1 = copy.deepcopy(graph)
    G2 = copy.deepcopy(graph)

    # Rename all nodes in G2 ---- USELESS
    mapping = {n: f"{n}'" for n in G2.nodes}
    G2 = nx.relabel_nodes(G2, mapping)

    # Shift positions
    for n in G2.nodes:
        if "pos" in G2.nodes:
            x, y = G2.nodes[n]["pos"]
            G2.nodes[n]["pos"] = (x + offset, y)
        else:
            G2.nodes[n]["pos"] = (offset, 0)

    # Create MultiGraph
    glued = nx.MultiGraph()
    glued.add_nodes_from(G1.nodes(data=True))
    glued.add_nodes_from(G2.nodes(data=True))
    glued.add_edges_from(G1.edges(data=True, keys=True))
    glued.add_edges_from(G2.edges(data=True, keys=True))

    # Identify open-end nodes (degree=1)
    open1 = [n for n in G1.nodes if G1.degree[n] == 1]
    open2 = [n for n in G2.nodes if G2.degree[n] == 1]

    open1.sort()
    open2.sort()

    # Glue open ends
    for n1, n2 in zip(open1, open2):
        neighbor1 = list(G1.neighbors(n1))[0]
        neighbor2 = list(G2.neighbors(n2))[0]

        # Check if the edge is connecting two open ends in both copies
        if G1.degree[neighbor1] == 1 and G2.degree[neighbor2] == 1:
            # Collapse into a single vertex with one edge
            label1 = G1[n1][neighbor1][0]["label"]
            glued.add_node(neighbor1, **G1.nodes[neighbor1])
            # Only add a single edge with the original label
            glued.add_edge(neighbor1, neighbor1, label=label1)
            # Remove the old open nodes and their edges
            glued.remove_node(n1)
            glued.remove_node(n2)
            continue

        # Use first edge label
        label1 = G1[n1][neighbor1][0]["label"]
        label2 = G2[n2][neighbor2][0]["label"]

        # Add edge connecting the two central vertices
        glued.add_edge(neighbor1, neighbor2, label=label1)
        print(f"Added edge {label1} between node {neighbor1} and node {neighbor2}")

        # Remove open-end edges
        glued.remove_edge(n1, neighbor1)
        print(f"Removed edge {label1} between {n1} and {neighbor1}")
        glued.remove_edge(n2, neighbor2)
        print(f"Removed edge {label2} between {n2} and {neighbor2}")

        # Remove open-end nodes
        glued.remove_node(n1)
        print(f"Removed node {n1}")
        glued.remove_node(n2)
        print(f"Removed node {n2}")

    return glued