import networkx as nx
from copy import deepcopy
from .utils import list_face_cycles, pick_smallest_interior_face_gt3, external_neighbor_in_trivalent, get_edge_label, add_edge_with_label, remove_one_cycle_edge, is_numeric_label, to_doubled, f_range_symbolic, uv_parallel_labels, incident_labels_excluding_pair
import matplotlib.pyplot as plt
from .drawing import draw_graph
import os

os.makedirs("graph_snapshots", exist_ok=True)

F_COUNTER = {"value": 0}

# --------------------
# Coefficient builders
# --------------------

def build_kronecker_coeff(c, d, power=1):
    """
    Kronecker recording the constraint c == d (stored even if equality is numeric and enforced).
    args order [c, d]; doubles included if numeric.
    """
    fixed = {"c": c, "d": d}
    for key in ["c", "d"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built Kronecker coeff with fixed labels:", c, d)
    return {
        "type": "Kronecker",
        "list_order": ["c", "d"],
        "fixed": fixed,
        "power": power
    }

def build_delta_coeff(j, power=1):
    """
    delta coefficient.
    """
    fixed = {"j": j}
    if is_numeric_label(j):
        fixed["J"] = to_doubled(j)
    print(f"Built delta coeff with fixed label:", j)
    return {
        "type": "delta",
        "list_order": ["j"],
        "fixed": fixed,
        "power": power
    }

def build_theta_coeff(a, b, c, power=1):
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
        "power": power
    }

def build_6j_coeff(a, b, f, c, d, e, power=1):
    fixed = {"a": a, "b": b, "f": f, "c": c, "d": d, "e": e}

    # doubled numeric values
    for key in ["a", "b", "c", "d", "e", "f"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built 6j coeff with fixed labels:", a, b, f, c, d, e)

    return {
        "type": "6j",
        "list_order": ["a","b","f","c","d","e"],
        "fixed": fixed,
        "power": power
    }

def build_sum_coeff(f, rng):
    return {
        "type": "sum",
        "index": f,      # e.g., "f_3"
        "range2": rng    # numeric doubled bounds or None
    }

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
            print(f"Found triangle candidate with nodes:", cycle, "and edges:", [get_edge_label(G, cycle[i], cycle[(i+1)%3]) for i in range(3)])
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

        # Record Kronecker(a,b)
        coeffs.append(build_kronecker_coeff(a, b))

        # Remove the vertex v (drops its two incident edges)
        if G.has_node(v):
            G.remove_node(v)

        # Add the new edge between n1 and n2 labeled a (choice irrelevant due to Kronecker)
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
                coeffs.append(build_delta_coeff(c))
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

    # Record Kronecker on the external edges
    coeffs.append(build_kronecker_coeff(a_lbl, b_lbl))

    # Record theta: internal edge + external edges
    coeffs.append(build_theta_coeff(a_lbl, int_lbl_1, int_lbl_2))
    coeffs.append(build_delta_coeff(a_lbl, -1))

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

    # External neighbors and labels (a,d,f)
    au_node = external_neighbor_in_trivalent(G, u, tri)
    dv_node = external_neighbor_in_trivalent(G, v, tri)
    fw_node = external_neighbor_in_trivalent(G, w, tri)

    if not all([au_node, dv_node, fw_node]):
        return None

    # Read labels before modifying the graph
    a = get_edge_label(G, u, au_node)
    b = get_edge_label(G, w, u)
    c = get_edge_label(G, v, w)
    d = get_edge_label(G, v, dv_node)
    e = get_edge_label(G, u, v)
    f = get_edge_label(G, w, fw_node)    

    # We will keep T = u and remove v, w
    T = w

    # Remove v and w (this drops their incident edges, including the triangle edges)
    if G.has_node(v):
        G.remove_node(v)
    if G.has_node(u):
        G.remove_node(u)

    # Reattach external edges that used to go to v and w, now to T
    # Edge (u, au_node) labeled a already exists; keep it.
    # Add edges (T, dv_node) with label d and (T, au_node) with label a
    add_edge_with_label(G, T, dv_node, d)
    add_edge_with_label(G, T, au_node, a)
    print(f"Collapsed triangle nodes:", u, v, w, "into node:", T)

    coeffs.append(build_6j_coeff(a, b, f, c, d, e))
    coeffs.append(build_theta_coeff(b, c, f))
    coeffs.append(build_delta_coeff(f, -1))

    print(f"Applied triangle reduction to : ", u, v, w)

    # Print graph
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

    F_COUNTER["value"] += 1
    f_symbol = f"F_{F_COUNTER['value']}"

    add_edge_with_label(G, u_node, v_node, f_symbol)

    # CRITICAL FIX: Always create summation coefficient for F-variables
    # Use f_range_with_symbolic to handle both numeric and symbolic edge labels
    from src.utils import f_range_with_symbolic

    # Get known ranges from existing F-variables (stored in term metadata)
    known_ranges = term.get("f_ranges", {})

    rng = f_range_with_symbolic(a_label, b_label, d_label, c_label, known_ranges=known_ranges)
    if rng is not None:
        coeffs.append(build_sum_coeff(f_symbol, rng))  # summation over f

        # Store range info for this F-variable
        if "f_ranges" not in term:
            term["f_ranges"] = {}
        term["f_ranges"][f_symbol] = (rng.get('Fmin'), rng.get('Fmax'))

        # Enhanced logging with range quality indicator
        range_j_min = rng.get('Fmin') // 2
        range_j_max = rng.get('Fmax') // 2
        range_size = range_j_max - range_j_min + 1
        quality = "tight" if range_size <= 5 else ("moderate" if range_size <= 15 else "wide")

        print(f"  → Created summation: {f_symbol} with range {range_j_min} to {range_j_max} " +
              f"[{range_size} values, {quality}]" +
              (" (symbolic)" if rng.get('symbolic') else ""))
    else:
        # Fallback: even if range computation fails, create summation with default range
        # This ensures F-variables always have summation symbols
        print(f"  ⚠️  Could not compute range for {f_symbol}, using conservative default [0, 20]")
        coeffs.append(build_sum_coeff(f_symbol, {"Fmin": 0, "Fmax": 40, "parity": 0}))

        if "f_ranges" not in term:
            term["f_ranges"] = {}
        term["f_ranges"][f_symbol] = (0, 40)

    # Record the 6j with correct mapping [a, b, f, d, c, e]
    coeffs.append(build_6j_coeff(a=a_label, b=b_label, f=f_symbol, c=c_label, d=d_label, e=e_label))

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

def reduce_all_cycles(glued_graph, animator=None):
    """
    Global reduction loop:
      - repeatedly apply local reductions,
      - if a cycle of length > 3 exists, apply a single F-move to shrink it,
      - repeat until no cycles > 3 remain and no changes occur.
    Returns a single final term.

    Parameters:
    -----------
    glued_graph : nx.MultiGraph
        The graph to reduce
    animator : ReductionAnimator, optional
        If provided, captures visualization steps during reduction
    """

    term = {"graph": glued_graph, "coeffs": []}

    # Capture initial state
    if animator:
        animator.add_step(
            term["graph"],
            title="Initial Glued Graph",
            description="Starting graph after gluing open edges. Ready to begin reduction.",
            operation="initial"
        )

    max_iters = 10000
    for _ in range(max_iters):

        #
        # ---- 1) LOCAL FIXPOINT CLEANUP ----
        #
        changed = False
        cleanup_count = 0
        while True:
            prev_sig   = graph_signature(term["graph"])
            prev_coeff = len(term["coeffs"])

            term = reduce_all_thetas(term)
            term = reduce_all_two_cycles(term)
            term = reduce_all_triangles(term)
            term = reduce_all_degree2(term)
            term = reduce_all_loops(term)

            # check if anything happened
            if graph_signature(term["graph"]) == prev_sig and len(term["coeffs"]) == prev_coeff:
                break
            changed = True
            cleanup_count += 1

            # Capture cleanup steps (but not every single one to avoid too many frames)
            if animator and cleanup_count % 3 == 0:
                animator.add_step(
                    term["graph"],
                    title=f"Local Reductions (pass {cleanup_count})",
                    description="Applying theta, 2-cycle, triangle, degree-2, and loop reductions.",
                    operation="triangle"
                )


        #
        # ---- 2) TRY TO APPLY AN F–MOVE ----
        #
        cycles = list_face_cycles(term["graph"])
        C = pick_smallest_interior_face_gt3(cycles) if cycles else None

        # no more large faces -> terminate if nothing else changes
        if not C or len(C) <= 3:
            if not changed:
                # Final state
                if animator:
                    animator.add_step(
                        term["graph"],
                        title="Final Reduced Graph",
                        description="All cycles reduced. Graph is now fully simplified!",
                        operation="triangle"
                    )
                return [term]
            continue

        # attempt one F–move
        new_term = f_move_recouple_term(term, C, i=0)

        if new_term is None:
            # face exists, but F–move not applicable -> we are stuck
            if animator:
                animator.add_step(
                    term["graph"],
                    title="Reduction Complete (stuck)",
                    description=f"Cannot apply F-move on face {C}. Reduction terminates here.",
                    highlight_nodes=list(C),
                    operation="f-move"
                )
            return [term]

        # Capture F-move step
        if animator:
            animator.add_step(
                new_term["graph"],
                title=f"F-move Applied on {len(C)}-cycle",
                description=f"Applied F-move recoupling on face {C}. Inserted new internal node and 6j symbol.",
                highlight_nodes=list(C),
                operation="f-move"
            )

        term = new_term

    # safety cap
    if animator:
        animator.add_step(
            term["graph"],
            title="Maximum Iterations Reached",
            description="Safety cap: Maximum iteration limit reached. Stopping reduction.",
            operation="triangle"
        )
    return [term]