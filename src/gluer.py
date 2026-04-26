import networkx as nx
import copy

# -----------------------------------------------------------------------
# Graph gluing: turn an open spin network into a closed one
# -----------------------------------------------------------------------
#
# A spin network drawn by the user has "open ends": nodes of degree 1 that
# represent external legs. To compute the norm we must close these legs by
# connecting the network to a copy of itself (the "bra" glued to the "ket").
#
# Concretely: given graph G, we build G1 = G and G2 = relabelled copy of G,
# then connect each open-end node of G1 to the matching open-end node of G2
# by merging their internal neighbours with a new edge (same label).
# The result is a closed trivalent graph called the "theta graph" (for a
# simple two-legged network it literally becomes a theta shape).

def glue_open_edges(graph, offset=10):
    G1 = copy.deepcopy(graph)
    G2 = copy.deepcopy(graph)

    # Relabel G2's nodes with a prime suffix to avoid collisions with G1.
    mapping = {n: f"{n}'" for n in G2.nodes}
    G2 = nx.relabel_nodes(G2, mapping)

    # Shift G2's stored positions so the two copies don't overlap visually.
    for n in G2.nodes:
        if "pos" in G2.nodes:
            x, y = G2.nodes[n]["pos"]
            G2.nodes[n]["pos"] = (x + offset, y)
        else:
            G2.nodes[n]["pos"] = (offset, 0)

    # Start the glued graph with all nodes and edges from both copies.
    glued = nx.MultiGraph()
    glued.add_nodes_from(G1.nodes(data=True))
    glued.add_nodes_from(G2.nodes(data=True))
    glued.add_edges_from(G1.edges(data=True, keys=True))
    glued.add_edges_from(G2.edges(data=True, keys=True))

    # Collect open-end nodes (degree 1) from G1, then derive the matching G2
    # nodes via the known relabelling n -> f"{n}'".  Sorting G2 independently
    # would fail when node IDs contain numbers >= 10 because lexicographic
    # order ("10'" < "6'") differs from numeric order (10 > 6).
    open1 = sorted([n for n in G1.nodes if G1.degree[n] == 1], key=str)
    open2 = [f"{n}'" for n in open1]

    # Pair each open end in G1 with the corresponding open end in G2.
    for n1, n2 in zip(open1, open2):
        neighbor1 = list(G1.neighbors(n1))[0]
        neighbor2 = list(G2.neighbors(n2))[0]

        # Special case: if both neighbours are also open ends (both degree 1),
        # the two legs form a single isolated edge. Collapse them into one
        # self-loop at the surviving node instead of connecting two open ends.
        if G1.degree[neighbor1] == 1 and G2.degree[neighbor2] == 1:
            label1 = G1[n1][neighbor1][0]["label"]
            glued.add_node(neighbor1, **G1.nodes[neighbor1])
            glued.add_edge(neighbor1, neighbor1, label=label1)
            glued.remove_node(n1)
            glued.remove_node(n2)
            continue

        # Normal case: add an edge between the two internal neighbours,
        # then remove the now-redundant open-end nodes and their stub edges.
        label1 = G1[n1][neighbor1][0]["label"]

        glued.add_edge(neighbor1, neighbor2, label=label1)

        glued.remove_edge(n1, neighbor1)
        glued.remove_edge(n2, neighbor2)
        glued.remove_node(n1)
        glued.remove_node(n2)

    return glued
