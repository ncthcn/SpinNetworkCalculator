import networkx as nx
from copy import deepcopy
from utils import list_face_cycles, pick_smallest_interior_face_gt3, external_neighbor_in_trivalent, get_edge_label, add_edge_with_label, remove_one_cycle_edge, is_numeric_label, to_doubled, f_range_symbolic, uv_parallel_labels, incident_labels_excluding_pair
import matplotlib.pyplot as plt
from drawing import draw_graph
import os

os.makedirs("graph_snapshots", exist_ok=True)

# step_counter = 0

sum_counter = 1

# def save_graph_snapshot(G, note=""):
#     global step_counter
#     edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
#     draw_graph(G, edge_labels)
#     plt.title(f"Step {step_counter}: {note}")
#     plt.savefig(f"graph_step_{step_counter}.png")
#     plt.close()
#     step_counter += 1

# --------------------
# Coefficient builders
# --------------------

def build_delta_coeff(c, d):
    """
    Kronecker delta recording the constraint c == d (stored even if equality is numeric and enforced).
    args order [c, d]; doubles included if numeric.
    """
    fixed = {"c": c, "d": d}
    for key in ["c", "d"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built delta coeff with fixed labels:", c, d)
    return {
        "type": "delta",
        "list_order": ["c", "d"],
        "fixed": fixed,
        "sum_index": None
    }

def build_theta_coeff(a, b, c):
    """
    Theta coefficient for 2-cycle collapse.
    Requires numeric equality c == d checked elsewhere; stores numeric doubles if available.
    """
    fixed = {"a": a, "b": b, "c": c}
    for key in ["a", "b", "c"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built theta coeff with fixed labels:", a, b, c)
    return {
        "type": "theta",
        "list_order": ["a", "b", "c"],
        "fixed": fixed,
        "sum_index": None
    }

def build_triangle_coeff(a, b, c, d, e, f):
    """
    Final triangle reduction {adf}+{bde}+{efc} -> {abc}; attach its 6j.
    Stored as a fixed (non-summed) coefficient for the last step.
    Works with numeric or symbolic labels.
    """
    fixed = {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}
    for key in ["a", "b", "c", "d", "e", "f"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built triangle coeff with fixed labels:", a, b, c, d, e, f)
    return {
        "type": "6j",
        "list_order": ["a", "b", "c", "d", "e", "f"],
        "fixed": fixed,
        "sum_index": None
    }

def build_fmove_coeff(a, b, d, c, e):
    """
    Package the coefficient entry for an F-move with args order [a, b, f, d, c, e].
    f is symbolic; we store its compact summation range (doubled) if inputs are numeric.
    """
    fixed = {"a": a, "b": b, "d": d, "c": c, "e": e}
    # Doubled forms if numeric, else leave absent
    for key in ["a", "b", "d", "c", "e"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)

    f_symbol = f"f_{sum_counter}"

    rng = f_range_symbolic(a, b, d, c)  # None if non-numeric or parity inconsistency

    coeff_entry = {
        "type": "6j",
        "list_order": ["a", "b", "f_{sum_counter}", "d", "c", "e"],
        "fixed": fixed,           # contains numeric doubles when available
        "sum_index": "f_{sum_counter}",
        "sum_range2": rng,        # in doubled form; use later to enumerate if needed
        # Generators to produce per-f args when you later enumerate:
        "gen_args": lambda f: [a, b, f_symbol, d, c, e],
        "gen_args2": lambda F: [
            fixed.get("A"), fixed.get("B"), F, fixed.get("D"), fixed.get("C"), fixed.get("E")
        ],
    }
    print(f"Built F-move coeff:", a, b, d, c, e, "with range:", rng)
    return coeff_entry


# ------------------
# Subnetwork finders
# ------------------

def find_two_cycle_candidate(G):
    """
    Find a reducible 2-cycle (digon) in the graph:
    - u, v connected by 2 parallel edges (internal edges),
    - each of u and v has exactly one neighbor outside the 2-cycle (external edge),
    Returns (u, v, a_label, b_label, internal_label, a_node, b_node)
    where:
        a_label = external edge of u,
        b_label = external edge of v,
        internal_label = label of the 2-cycle edge to keep.
    """
    simple = nx.Graph(G)
    for u, v in simple.edges():
        labels_uv = uv_parallel_labels(G, u, v)
        if len(labels_uv) != 2:
            continue
        print(f"Trying digon:", u, v, labels_uv)
        # Check external neighbors
        ext_u = incident_labels_excluding_pair(G, u, v)
        ext_v = incident_labels_excluding_pair(G, v, u)

        if len(ext_u) != 1 or len(ext_v) != 1:
            print(f"External neighbor count failed")
            continue

        a_lbl, a_node = ext_u[0]
        b_lbl, b_node = ext_v[0]

        if a_node is None or b_node is None:
            continue
        print(f"Found digon candidate. Vertex ", u, " with edges ", a_lbl, labels_uv[0], labels_uv[1], " and vertex ", v, " with edges ", b_lbl, labels_uv[0], labels_uv[1])
        return (a_node, u, v, b_node, a_lbl, labels_uv[0], labels_uv[1], b_lbl)
    return None

def find_theta_candidate(G):
    """
    Find a reducible theta in the graph:
    - u, v connected by 3 parallel edges,
    Returns (u, v, internal_lbl_1, internal_lbl_2, internal_lbl_3)
    """
    simple = nx.Graph(G)
    for u, v in simple.edges():
        labels_uv = uv_parallel_labels(G, u, v)
        if len(labels_uv) != 3:
            continue
        print(f"Found theta. Vertices ", u, " and ", v, " with edges ", labels_uv[0], labels_uv[1], labels_uv[2])
        return (u, v, labels_uv[0], labels_uv[1], labels_uv[2])
    return None

def find_triangle_candidate(G):
    """
    Find a reducible triangle in the graph:
    - u, v, w connected in a triangle (3-cycle),
    Returns (u, v, w)
    """
    cycles = list_face_cycles(G)
    for cycle in cycles:
        if len(cycle) == 3:
            print(f"Found triangle candidate with nodes:", cycle)
            return cycle
    return None

# ---------
# Reducers
# ---------

def apply_degree2_reduction(term):
    """
    If a vertex v has exactly two incident edges labeled a and b, impose delta(a,b),
    remove v, and connect its two neighbors with a single edge labeled a.
    Returns a new term if a reduction was performed, else None.
    """
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    for v in list(G.nodes()):
        # Collect exactly two incident edges on v (with keys and data)
        inc = list(G.edges(v, keys=True, data=True))
        if len(inc) != 2:
            continue

        # Each entry: (v, nbr, key, data) or (nbr, v, key, data), NetworkX normalizes as (u, v, k, data)
        (u1, w1, k1, d1) = inc[0]
        (u2, w2, k2, d2) = inc[1]

        # Find the other endpoints for the two edges (neighbors can be equal)
        n1 = w1 if u1 == v else u1
        n2 = w2 if u2 == v else u2

        a = d1.get("label", None)
        b = d2.get("label", None)

        # Record delta(a,b)
        coeffs.append(build_delta_coeff(a, b))

        # Remove the vertex v (drops its two incident edges)
        if G.has_node(v):
            G.remove_node(v)

        # Add the new edge between n1 and n2 labeled a (choice irrelevant due to delta)
        add_edge_with_label(G, n1, n2, a)
        print(f"Applied degree-2 reduction on node:", v)
        # save_graph_snapshot(term["graph"], note="after degree-2 reduction")
        edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
        draw_graph(term["graph"], EdgeLabels=edge_labels, note="after degree-2")
        return {"graph": G, "coeffs": coeffs}

    return None

def apply_loop_reduction(term):
    """
    Remove a self-loop (u,u) labeled c and record theta(c,c,0).
    Returns a new term if a loop was removed, else None.
    """
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    # Scan for any self-loop
    for u in list(G.nodes()):
        if u in G and u in G[u]:
            # There may be multiple loop edges; remove them one by one
            for k in list(G[u][u].keys()):
                data = G[u][u][k]
                c = data.get("label", None)
                # Record theta(c,c,0)
                coeffs.append(build_theta_coeff(c, c, 0))
                # Remove this loop edge
                G.remove_edge(u, u, key=k)
                print(f"Applied loop reduction on node:", u, "with label:", c)
                # save_graph_snapshot(term["graph"], note="after loop reduction")
                edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
                draw_graph(term["graph"], EdgeLabels=edge_labels, note="after loop")
                return {"graph": G, "coeffs": coeffs}

    return None

def apply_two_cycle_reduction(term):
    """
    Apply one 2-cycle (digon) reduction:
    - δ(external_u, external_v) records equality of the two external edges,
    - θ(external_u, external_v, internal) for the 2-cycle,
    - remove u, v and replace with single edge between their external neighbors.
    """
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    cand = find_two_cycle_candidate(G)
    if cand is None:
        return None

    a_node, u, v, b_node, a_lbl, int_lbl_1, int_lbl_2, b_lbl = cand

    # Record delta on the external edges
    coeffs.append(build_delta_coeff(a_lbl, b_lbl))

    # Record theta: internal edge + external edges
    coeffs.append(build_theta_coeff(a_lbl, int_lbl_1, int_lbl_2))

    # Remove u, v
    if G.has_node(u):
        G.remove_node(u)
    if G.has_node(v):
        G.remove_node(v)
    if G.has_edge(u, v):
        G.remove_edge(u, v)
    if G.has_edge(a_node, u):
        G.remove_edge(a_node, u)
    if G.has_edge(b_node, v):
        G.remove_edge(b_node, v)

    # Connect the external neighbors with the internal label
    add_edge_with_label(G, a_node, b_node, a_lbl)
    print(f"Applied 2-cycle reduction on nodes:", u, v)
    # save_graph_snapshot(term["graph"], note="after 2-cycle reduction")
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    draw_graph(term["graph"], EdgeLabels=edge_labels, note="after 2-cycle")
    return {"graph": G, "coeffs": coeffs}

def apply_theta_reduction(term):
    """
    Apply one theta reduction:
    - θ(internal_lbl_1, internal_lbl_2, internal_lbl_3) for the theta,
    - remove u, v and the 3 edges between them.
    """
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    cand = find_theta_candidate(G)
    if cand is None:
        return None

    u, v, int_lbl_1, int_lbl_2, int_lbl_3 = cand

    # Record theta: internal edge + external edges
    coeffs.append(build_theta_coeff(int_lbl_1, int_lbl_2, int_lbl_3))

    # Remove u, v
    if G.has_node(u):
        G.remove_node(u)
    if G.has_node(v):
        G.remove_node(v)
    if G.has_edge(u, v):
        G.remove_edge(u, v)
    print(f"Applied theta reduction on nodes:", u, v)
    # save_graph_snapshot(term["graph"], note="after theta reduction")
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    draw_graph(term["graph"], EdgeLabels=edge_labels, note="after theta")
    return {"graph": G, "coeffs": coeffs}

def apply_triangle_reduction(term):
    """
    Apply one triangle reduction:
    - collapse triangle topology and record the 6j,
    - remove u, v, w and reattach external edges.
    """
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    tri = find_triangle_candidate(G)
    if tri is None:
        return None
    
    print(f"Reducing triangle: ", tri)
    
    u = tri[0]
    v = tri[1]
    w = tri[2]

    # External neighbors and labels (a,b,c)
    au_node = external_neighbor_in_trivalent(G, u, tri)
    bv_node = external_neighbor_in_trivalent(G, v, tri)
    cw_node = external_neighbor_in_trivalent(G, w, tri)

    if not all([au_node, bv_node, cw_node]):
        return None

    # Read labels before modifying the graph
    a = get_edge_label(G, u, au_node)
    b = get_edge_label(G, v, bv_node)
    c = get_edge_label(G, w, cw_node)
    d = get_edge_label(G, u, v)
    e = get_edge_label(G, v, w)
    f = get_edge_label(G, w, u)

    # We will keep T = u and remove v, w
    T = u

    # Remove v and w (this drops their incident edges, including the triangle edges)
    if G.has_node(v):
        G.remove_node(v)
    if G.has_node(w):
        G.remove_node(w)

    # Reattach external edges that used to go to v and w, now to T
    # Edge (u, au_node) labeled a already exists; keep it.
    # Add edges (T, bv_node) with label b and (T, cw_node) with label c
    add_edge_with_label(G, T, bv_node, b)
    add_edge_with_label(G, T, cw_node, c)
    print(f"Collapsed triangle nodes:", u, v, w, "into node:", T)

    coeffs.append(build_triangle_coeff(a, b, c, d, e, f))

    print(f"Applied triangle reduction on nodes:", u, v, w)
    # save_graph_snapshot(term["graph"], note="after triangle reduction")
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    draw_graph(term["graph"], EdgeLabels=edge_labels, note="after triangle")

    return {"graph": G, "coeffs": coeffs}

# -----------------------------------------------
# F-move application (reducing cycle length by 1)
# -----------------------------------------------

def f_move_recouple_term(term, cycle_nodes, i):
    """
    Apply one F-move on consecutive cycle vertices:
    b_node = cycle[i], d_node = cycle[(i+1) % N], e = label of (b_node,d_node) on the cycle.
    External neighbors: a_node (at b_node), c_node (at d_node).
    Correct mapping to args [a, b, f, d, c, e]:
      - a = label(b_node, a_node)  external at b_node
      - b = label(b_node, prev_b)  cycle at b_node
      - e = label(b_node, d_node)  cycle edge between the pair
      - d = label(d_node, next_d)  cycle at d_node
      - c = label(d_node, c_node)  external at d_node
    Topology: remove (b_node,d_node); add diagonal (prev_b, next_d) labeled f_symbol.
    """
    print(f"Applying F-move on cycle nodes:", cycle_nodes, "at index:", i)
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    N = len(cycle_nodes)
    b_node = cycle_nodes[(i - 1) % N]
    u_node = cycle_nodes[i % N]
    v_node = cycle_nodes[(i + 1) % N]
    c_node = cycle_nodes[(i + 2) % N]


    # External neighbors (unique in trivalent)
    a_node = external_neighbor_in_trivalent(G, u_node, cycle_nodes)
    d_node = external_neighbor_in_trivalent(G, v_node, cycle_nodes)
    if a_node is None or d_node is None:
        return None

    # Edge labels
    a_label  = get_edge_label(G, u_node, a_node)    # external edge a
    b_label  = get_edge_label(G, u_node, b_node)    # cycle edge b
    c_label  = get_edge_label(G, v_node, c_node)    # cycle edge c
    d_label  = get_edge_label(G, v_node, d_node)    # external edge d
    e_label  = get_edge_label(G, u_node, v_node)    # cycle edge e

    if any(x is None for x in [e_label, a_label, b_label, d_label, c_label]):
        return None

    # Topology change: remove old cycle edge and add new diagonal f
    remove_one_cycle_edge(G, u_node, a_node)
    remove_one_cycle_edge(G, u_node, v_node)
    remove_one_cycle_edge(G, v_node, c_node)
    add_edge_with_label(G, c_node, u_node, c_label)
    add_edge_with_label(G, a_node, v_node, a_label)

    f_symbol = f"f_{sum_counter}"
    sum_counter = sum_counter + 1
    add_edge_with_label(G, u_node, v_node, f_symbol)

    # Record the 6j with correct mapping [a, b, f, d, c, e]
    coeffs.append(build_fmove_coeff(a=a_label, b=b_label, d=d_label, c=c_label, e=e_label))

    print(f"Applied F-move on nodes:", b_node, d_node, "with new diagonal f:", f_symbol)
    edge_labels = {(u, v, k): d["label"] for u, v, k, d in G.edges(keys=True, data=True)}
    draw_graph(term["graph"], EdgeLabels=edge_labels, note="after F-move")
    return {"graph": G, "coeffs": coeffs}

# -------------
# Bulk reducers
# -------------

def reduce_all_degree2(term):
    """
    Repeatedly apply degree-2 reductions until none remain.
    """
    cur = term
    while True:
        t2 = apply_degree2_reduction(cur)
        if t2 is None:
            return cur
        cur = t2

def reduce_all_loops(term):
    """
    Repeatedly remove all self-loops, recording theta(c,c,0) for each.
    """
    cur = term
    while True:
        t2 = apply_loop_reduction(cur)
        if t2 is None:
            return cur
        cur = t2

def reduce_all_two_cycles(term):
    """
    Repeatedly reduce all thetas until none remain.
    """
    cur = term
    while True:
        t2 = apply_two_cycle_reduction(cur)
        if t2 is None:
            print(f"No more 2-cycles detected.")
            break  # no more 2-cycles
        cur = t2
    return cur

def reduce_all_thetas(term):
    """
    Repeatedly reduce all thetas until none remain.
    """
    cur = term
    while True:
        t2 = apply_theta_reduction(cur)
        if t2 is None:
            print(f"No more thetas detected.")
            break  # no more thetas
        cur = t2
    return cur

def reduce_all_triangles(term):
    """
    Repeatedly reduce all triangles until none remain.
    """
    cur = term
    while True:
        t2 = apply_triangle_reduction(cur)
        if t2 is None:
            print(f"No more triangles detected.")
            break  # no more triangles
        cur = t2
    return cur

# -----------------------------------------------------------------------
# Reduce N-cycle to triangles, then triangle to single vertex with {a,b,c}
# -----------------------------------------------------------------------

def reduce_cycle_to_triangle(initial_term, cycle_nodes):
    """
    Reduce a cycle to a triangle by repeatedly applying F-moves,
    each time recomputing the current largest cycle from the updated graph.
    Then collapse the triangle topology and record the 6j.
    """
    print(f"Reducing cycle to triangle for nodes:", cycle_nodes)
    terms = [initial_term]

    # Keep applying F-moves until the target cycle shrinks to 3
    for _ in range(1000):  # safety cap
        new_terms = []
        progressed = False
        for term in terms:
            # Recompute the current cycle in this term
            cycles = list_face_cycles(term["graph"])
            C = pick_smallest_interior_face_gt3(cycles) if cycles else None
            if not C or len(C) <= 3:
                # No F-move needed for this term
                new_terms.append(term)
                continue

            # Apply one F-move on a consecutive pair (pick i=0)
            t2 = f_move_recouple_term(term, C, i=0)
            if t2 is not None:
                new_terms.append(t2)
                progressed = True
            else:
                # Couldn’t apply—keep term
                new_terms.append(term)

        terms = new_terms
        if not progressed:
            break  # nothing more to do

    # Now collapse triangles and attach 6j's
    finalized = []
    for term in terms:
        G = deepcopy(term["graph"])
        coeffs = term["coeffs"][:]

        tri = pick_smallest_interior_face_gt3(list_face_cycles(G))
        if not tri or len(tri) != 3:
            finalized.append(term)
            continue

        u, v, w = tri
        collapsed = collapse_triangle_topology(G, u, v, w)
        if collapsed is None:
            finalized.append(term)
            continue

        G2, a, b, c, d, e, f = collapsed
        coeffs.append(build_triangle_coeff(a, b, c, d, e, f))

        # Clean up trivial structures formed by the collapse
        new_term = {"graph": G2, "coeffs": coeffs}
        new_term = reduce_all_loops(new_term)
        new_term = reduce_all_degree2(new_term)
        new_term = reduce_all_two_cycles(new_term)

        finalized.append(new_term)
    print(f"Reduced cycle to triangle and collapsed it.")
    return finalized

# ----------------------------------------------------------------
# Global reduction: keep reducing while cycles of length > 2 exist
# ----------------------------------------------------------------

def graph_signature(G):
    """
    Canonical signature of a multigraph without sorting or string conversion:
    - Each edge represented as (frozenset of endpoints, label, key)
    - Parallel edges distinguished by keys
    - Returns a frozenset of these tuples, suitable for equality checks
    """
    sig = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        lbl = data.get("label", None)
        endpoints = frozenset([u, v])  # keep original node objects
        sig.add((endpoints, lbl, k))   # include key to distinguish parallel edges
    return frozenset(sig)


def reduce_all_cycles(glued_graph):
    """
    Fixpoint global reduction:
    - repeatedly run local reductions (loops, degree-2, digons),
    - then, if a cycle of length > 2 exists, reduce it to a triangle and collapse,
    - repeat until no changes occur.
    Returns a single term (graph, coeffs).
    """
    term = {"graph": glued_graph, "coeffs": []}

    max_iters = 10000  # safety cap to avoid accidental infinite loops
    for _ in range(max_iters):
        changed = False
        prev_sig = graph_signature(term["graph"])
        prev_coeff_count = len(term["coeffs"])

        # Local cleanups to fixpoint
        t = reduce_all_thetas(term)
        t = reduce_all_two_cycles(t)
        t = reduce_all_triangles(t)
        t = reduce_all_degree2(t)
        t = reduce_all_loops(t)

        # Check for local-change progress
        if graph_signature(t["graph"]) != prev_sig or len(t["coeffs"]) != prev_coeff_count:
            changed = True
        term = t

        # Check if any cycles remain
        cycles = list_face_cycles(term["graph"])
        C = pick_smallest_interior_face_gt3(cycles) if cycles else None

        if not C or len(C) <= 2:
            # No big cycles left; if no local changes happened, we are done
            if not changed:
                return [term]
            # Otherwise, loop again to see if more local reductions are possible
            continue

        # Reduce one big cycle to a triangle and collapse
        reduced_terms = reduce_cycle_to_triangle(term, C)
        # We expect a single term; if multiple, take the first (we keep f symbolic)
        term = reduced_terms[0] if reduced_terms else term

        # After a batch, run local cleanups again next iteration
        # The loop continues; each triangle collapse reduces edges, guaranteeing progress

    # If we ever hit max_iters, return what we have (shouldn’t happen in well-formed inputs)
    return [term]