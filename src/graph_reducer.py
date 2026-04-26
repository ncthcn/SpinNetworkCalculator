import networkx as nx
from copy import deepcopy
from .utils import (list_face_cycles, pick_smallest_interior_face_gt3,
                    external_neighbor_in_trivalent, get_edge_label,
                    add_edge_with_label, remove_one_cycle_edge,
                    is_numeric_label, to_doubled, f_range_symbolic,
                    uv_parallel_labels, incident_labels_excluding_pair,
                    f_range_with_symbolic)
import os

os.makedirs("graph_snapshots", exist_ok=True)

# Global counter for generating unique F-variable names (F_1, F_2, …).
# Each F-move introduces a new summation index; the counter ensures names
# never clash across multiple F-moves in one reduction.
F_COUNTER = {"value": 0}

# -----------------------------------------------------------------------
# Coefficient builders
# -----------------------------------------------------------------------
#
# Reductions replace graph substructures with algebraic factors. Each factor
# is stored as a Python dict with a "type" field. The builders below create
# these dicts in a uniform format consumed by norm_reducer.py and
# spin_evaluator.py.
#
# Numeric labels are stored both as-is and in "doubled" form (key.upper()),
# because wigxjpf (the 6j backend) requires 2*j integers.

# Records the constraint that spin label c must equal spin label d.
# Arises when two edges that must carry the same spin are merged;
# apply_kroneckers() later substitutes one for the other.
def build_kronecker_coeff(c, d, power=1):
    fixed = {"c": c, "d": d}
    for key in ["c", "d"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    return {
        "type": "Kronecker",
        "list_order": ["c", "d"],
        "fixed": fixed,
        "power": power
    }

# Records a factor Δ_j = (-1)^(2j) (2j+1).
# Appears when a self-loop of spin j is removed (loop reduction) or as a
# factor in 2-cycle and triangle reductions.
def build_delta_coeff(j, power=1):
    fixed = {"j": j}
    if is_numeric_label(j):
        fixed["J"] = to_doubled(j)
    return {
        "type": "delta",
        "list_order": ["j"],
        "fixed": fixed,
        "power": power
    }

# Records a factor Θ(a, b, c) = (a+b+c+1)! / [(a+b-c)!(a-b+c)!(-a+b+c)!].
# Appears in 2-cycle reductions (one theta per digon) and triangle
# reductions (one theta for the internal triangle).
def build_theta_coeff(a, b, c, power=1):
    fixed = {"a": a, "b": b, "c": c}
    for key in ["a", "b", "c"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    return {
        "type": "theta",
        "list_order": ["a", "b", "c"],
        "fixed": fixed,
        "power": power
    }

# Records a standard (pre-expansion) 6j symbol {a b f; c d e}.
# Created by both triangle reductions and F-moves. Later expanded by
# expand_6j_symbolic() into Wigner 6j form with theta/delta prefactors.
def build_6j_coeff(a, b, f, c, d, e, power=1):
    fixed = {"a": a, "b": b, "f": f, "c": c, "d": d, "e": e}
    for key in ["a", "b", "c", "d", "e", "f"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    return {
        "type": "6j",
        "list_order": ["a", "b", "f", "c", "d", "e"],
        "fixed": fixed,
        "power": power
    }

# Records a summation ∑_{f=Fmin/2}^{Fmax/2} over the intermediate spin f.
# 'rng' is a dict with Fmin, Fmax in doubled units, and a parity flag.
# Created by f_move_recouple_term; evaluated as a loop in spin_evaluator.py.
def build_sum_coeff(f, rng):
    return {
        "type": "sum",
        "index": f,
        "range2": rng
    }

# -----------------------------------------------------------------------
# Subnetwork finders
# -----------------------------------------------------------------------
#
# Before applying a reduction, we scan the graph for a matchable substructure.
# Each finder returns enough information for the corresponding reducer.

# Looks for a 2-cycle (digon): two nodes u, v connected by exactly 2 parallel
# edges, each with exactly one external neighbour.
# Returns (a_node, u, v, b_node, a_lbl, int_lbl_1, int_lbl_2, b_lbl), or None.
def find_two_cycle_candidate(G):
    simple = nx.Graph(G)
    for u, v in simple.edges():
        labels_uv = uv_parallel_labels(G, u, v)
        if len(labels_uv) != 2:
            continue
        ext_u = incident_labels_excluding_pair(G, u, v)
        ext_v = incident_labels_excluding_pair(G, v, u)

        if len(ext_u) != 1 or len(ext_v) != 1:
            continue

        a_lbl, a_node = ext_u[0]
        b_lbl, b_node = ext_v[0]

        if a_node is None or b_node is None:
            continue
        return (a_node, u, v, b_node, a_lbl, labels_uv[0], labels_uv[1], b_lbl)
    return None

# Looks for a theta-graph: two nodes connected by exactly 3 parallel edges
# (and no other neighbours). This is the fully-reduced theta topology.
# Returns (u, v, lbl1, lbl2, lbl3), or None.
def find_theta_candidate(G):
    simple = nx.Graph(G)
    for u, v in simple.edges():
        labels_uv = uv_parallel_labels(G, u, v)
        if len(labels_uv) != 3:
            continue
        return (u, v, labels_uv[0], labels_uv[1], labels_uv[2])
    return None

# Looks for a triangle: a 3-cycle face in the planar embedding.
# Returns a list of 3 node ids, or None.
def find_triangle_candidate(G):
    cycles = list_face_cycles(G)
    for cycle in cycles:
        if len(cycle) == 3:
            return cycle
    return None

# -----------------------------------------------------------------------
# Individual reducers
# -----------------------------------------------------------------------
#
# Each reducer takes a 'term' dict {graph, coeffs} and returns a new term
# with the substructure removed and the corresponding coefficient appended,
# or None if the reduction is not applicable.
# We deepcopy the graph before modifying it so the original term is untouched.

# Degree-2 reduction: if vertex v has exactly 2 incident edges labeled a and b,
# then SU(2) recoupling forces a = b (Kronecker delta), so we remove v and
# connect its two neighbours with a single edge labeled a.
def apply_degree2_reduction(term):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    for v in list(G.nodes()):
        inc = list(G.edges(v, keys=True, data=True))
        if len(inc) != 2:
            continue

        (u1, w1, k1, d1) = inc[0]
        (u2, w2, k2, d2) = inc[1]

        n1 = w1 if u1 == v else u1
        n2 = w2 if u2 == v else u2

        a = d1.get("label", None)
        b = d2.get("label", None)

        coeffs.append(build_kronecker_coeff(a, b))

        if G.has_node(v):
            G.remove_node(v)

        add_edge_with_label(G, n1, n2, a)
        return {"graph": G, "coeffs": coeffs}

    return None

# Loop reduction: a self-loop on node u with spin label c evaluates to Δ_c.
# Remove the loop and record the delta factor.
def apply_loop_reduction(term):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    for u in list(G.nodes()):
        if u in G and u in G[u]:
            for k in list(G[u][u].keys()):
                data = G[u][u][k]
                c = data.get("label", None)
                coeffs.append(build_delta_coeff(c))
                G.remove_edge(u, u, key=k)
                return {"graph": G, "coeffs": coeffs}

    return None

# 2-cycle (digon) reduction: two nodes u, v with 2 parallel internal edges
# and one external edge each. The recoupling identity gives:
#   Kronecker(a, b) × Θ(a, int1, int2) × Δ_a^(-1)
# where a and b are the external labels and int1, int2 are the internal ones.
# Replace the digon by a single edge between the external neighbours.
def apply_two_cycle_reduction(term):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    cand = find_two_cycle_candidate(G)
    if cand is None:
        return None

    a_node, u, v, b_node, a_lbl, int_lbl_1, int_lbl_2, b_lbl = cand

    coeffs.append(build_kronecker_coeff(a_lbl, b_lbl))
    coeffs.append(build_theta_coeff(a_lbl, int_lbl_1, int_lbl_2))
    coeffs.append(build_delta_coeff(a_lbl, -1))

    if G.has_node(u):
        G.remove_node(u)
    if G.has_node(v):
        G.remove_node(v)
    # remove_node already drops incident edges, but guard in case of parallel edges
    for pair in [(u, v), (a_node, u), (b_node, v)]:
        if G.has_edge(*pair):
            G.remove_edge(*pair)

    add_edge_with_label(G, a_node, b_node, a_lbl)
    return {"graph": G, "coeffs": coeffs}

# Theta reduction: two nodes u, v connected by exactly 3 parallel edges.
# The theta evaluation identity gives: Θ(lbl1, lbl2, lbl3).
# Remove both nodes (and all 3 edges between them).
def apply_theta_reduction(term):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    cand = find_theta_candidate(G)
    if cand is None:
        return None

    u, v, int_lbl_1, int_lbl_2, int_lbl_3 = cand

    coeffs.append(build_theta_coeff(int_lbl_1, int_lbl_2, int_lbl_3))

    if G.has_node(u):
        G.remove_node(u)
    if G.has_node(v):
        G.remove_node(v)
    if G.has_edge(u, v):
        G.remove_edge(u, v)
    return {"graph": G, "coeffs": coeffs}

# Triangle reduction: three nodes u, v, w forming a 3-cycle, each with one
# external edge. The recoupling identity replaces this topology with a 6j
# symbol plus a theta and a delta:
#   {a b f; c d e}  ×  Θ(b, c, f)  ×  Δ_f^(-1)
# The three cycle nodes are removed; the three external edges are re-attached
# to a single surviving node (w) so the graph gains one node fewer.
def apply_triangle_reduction(term):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]
    tri = find_triangle_candidate(G)
    if tri is None:
        return None

    u, v, w = tri[0], tri[1], tri[2]

    au_node = external_neighbor_in_trivalent(G, u, tri)
    dv_node = external_neighbor_in_trivalent(G, v, tri)
    fw_node = external_neighbor_in_trivalent(G, w, tri)

    if not all([au_node, dv_node, fw_node]):
        return None

    # Read all labels before any graph modification.
    a = get_edge_label(G, u, au_node)   # external at u
    b = get_edge_label(G, w, u)         # triangle edge w-u
    c = get_edge_label(G, v, w)         # triangle edge v-w
    d = get_edge_label(G, v, dv_node)   # external at v
    e = get_edge_label(G, u, v)         # triangle edge u-v
    f = get_edge_label(G, w, fw_node)   # external at w

    # Keep w as the surviving node; remove u and v (and their triangle edges).
    T = w
    if G.has_node(v):
        G.remove_node(v)
    if G.has_node(u):
        G.remove_node(u)

    # Re-attach the external edges of u and v to T.
    add_edge_with_label(G, T, dv_node, d)
    add_edge_with_label(G, T, au_node, a)

    coeffs.append(build_6j_coeff(a, b, f, c, d, e))
    coeffs.append(build_theta_coeff(b, c, f))
    coeffs.append(build_delta_coeff(f, -1))

    return {"graph": G, "coeffs": coeffs}

# -----------------------------------------------------------------------
# F-move (recoupling move)
# -----------------------------------------------------------------------
#
# An F-move reduces a face of length n to a face of length n-1 by inserting
# a new internal edge labeled by a fresh summation variable F_k. This is the
# spin-network analogue of a 2→2 Pachner move.
#
# On the cycle, pick two adjacent vertices u_node and v_node (at positions i
# and i+1). Their external neighbours a_node and d_node are reconnected
# across the cycle; the old edge (u_node, v_node) is replaced by a new edge
# labeled F_k. This inserts a 6j symbol and a summation over F_k.
#
# The face length shrinks by 1 each time. Repeated F-moves reduce every face
# to a triangle, after which triangle reductions take over.

def f_move_recouple_term(term, cycle_nodes, i):
    G = deepcopy(term["graph"])
    coeffs = term["coeffs"][:]

    N = len(cycle_nodes)
    b_node = cycle_nodes[(i - 1) % N]
    u_node = cycle_nodes[i % N]
    v_node = cycle_nodes[(i + 1) % N]
    c_node = cycle_nodes[(i + 2) % N]

    # External neighbours of the two chosen cycle vertices.
    a_node = external_neighbor_in_trivalent(G, u_node, cycle_nodes)
    d_node = external_neighbor_in_trivalent(G, v_node, cycle_nodes)
    if a_node is None or d_node is None:
        return None

    # Read edge labels before modifying the graph.
    a_label = get_edge_label(G, u_node, a_node)   # external edge at u
    b_label = get_edge_label(G, u_node, b_node)   # cycle edge behind u
    c_label = get_edge_label(G, v_node, c_node)   # cycle edge ahead of v
    d_label = get_edge_label(G, v_node, d_node)   # external edge at v
    e_label = get_edge_label(G, u_node, v_node)   # cycle edge between u and v

    if any(x is None for x in [e_label, a_label, b_label, d_label, c_label]):
        return None

    # Topology surgery: rewire the cycle so the two external edges cross the
    # former (u,v) edge position, then insert the new F_k diagonal.
    remove_one_cycle_edge(G, u_node, a_node)
    remove_one_cycle_edge(G, u_node, v_node)
    remove_one_cycle_edge(G, v_node, c_node)
    add_edge_with_label(G, c_node, u_node, c_label)
    add_edge_with_label(G, a_node, v_node, a_label)

    F_COUNTER["value"] += 1
    f_symbol = f"F_{F_COUNTER['value']}"
    add_edge_with_label(G, u_node, v_node, f_symbol)

    # Compute the allowed range for F_k using both triangle inequalities.
    known_ranges = term.get("f_ranges", {})
    rng = f_range_with_symbolic(a_label, b_label, d_label, c_label,
                                known_ranges=known_ranges)

    if rng is not None:
        coeffs.append(build_sum_coeff(f_symbol, rng))
        if "f_ranges" not in term:
            term["f_ranges"] = {}
        term["f_ranges"][f_symbol] = (rng.get('Fmin'), rng.get('Fmax'))
    else:
        # Fall back to a conservative range when bounds cannot be computed.
        coeffs.append(build_sum_coeff(f_symbol, {"Fmin": 0, "Fmax": 40, "parity": 0}))
        if "f_ranges" not in term:
            term["f_ranges"] = {}
        term["f_ranges"][f_symbol] = (0, 40)

    # The F-move produces a 6j symbol with arguments [a, b, F, d, c, e].
    coeffs.append(build_6j_coeff(a=a_label, b=b_label, f=f_symbol,
                                 c=c_label, d=d_label, e=e_label))

    return {"graph": G, "coeffs": coeffs}

# -----------------------------------------------------------------------
# Bulk (fixpoint) reducers
# -----------------------------------------------------------------------
#
# Each function below runs its single-step reducer in a loop until it
# returns None (no more applicable substructures).

# Repeatedly removes degree-2 vertices until none remain.
def reduce_all_degree2(term):
    cur = term
    while True:
        t2 = apply_degree2_reduction(cur)
        if t2 is None:
            return cur
        cur = t2

# Repeatedly removes self-loops until none remain.
def reduce_all_loops(term):
    cur = term
    while True:
        t2 = apply_loop_reduction(cur)
        if t2 is None:
            return cur
        cur = t2

# Repeatedly collapses digons until none remain.
def reduce_all_two_cycles(term):
    cur = term
    while True:
        t2 = apply_two_cycle_reduction(cur)
        if t2 is None:
            break
        cur = t2
    return cur

# Repeatedly evaluates theta subgraphs until none remain.
def reduce_all_thetas(term):
    cur = term
    while True:
        t2 = apply_theta_reduction(cur)
        if t2 is None:
            break
        cur = t2
    return cur

# Repeatedly collapses triangles until none remain.
def reduce_all_triangles(term):
    cur = term
    while True:
        t2 = apply_triangle_reduction(cur)
        if t2 is None:
            break
        cur = t2
    return cur

# -----------------------------------------------------------------------
# Graph signature (for change detection)
# -----------------------------------------------------------------------

# A frozenset-based fingerprint of the multigraph used to detect whether any
# reduction step actually changed the graph. If both the signature and the
# coefficient list are unchanged after a full cleanup pass, the reduction has
# reached a fixpoint.
def graph_signature(G):
    sig = set()
    for u, v, k, data in G.edges(keys=True, data=True):
        lbl = data.get("label", None)
        endpoints = frozenset([u, v])
        sig.add((endpoints, lbl, k))
    return frozenset(sig)

# -----------------------------------------------------------------------
# Global reduction loop
# -----------------------------------------------------------------------

# Main entry point for the reduction pipeline. Starting from the glued (closed)
# graph, it alternates between:
#   1. A local fixpoint: apply all reductions (theta, 2-cycle, triangle, degree-2,
#      loop) until nothing changes.
#   2. A single F-move on the smallest remaining face of length > 3.
# This continues until no faces of length > 3 remain and no more local
# reductions apply — the graph is then fully reduced to a product of
# algebraic coefficients.
#
# 'animator' is optional; if provided, it receives a snapshot after each
# significant step so that a GIF/PDF can be produced.
# Returns a single-element list [term] for compatibility with callers that
# expect a list of terms (the reduction is always deterministic here).
def reduce_all_cycles(glued_graph, animator=None):
    term = {"graph": glued_graph, "coeffs": []}

    if animator:
        animator.add_step(
            term["graph"],
            title="Initial Glued Graph",
            description="Starting graph after gluing open edges. Ready to begin reduction.",
            operation="initial"
        )

    max_iters = 10000
    for _ in range(max_iters):

        # --- 1) LOCAL FIXPOINT ---
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

            if graph_signature(term["graph"]) == prev_sig and len(term["coeffs"]) == prev_coeff:
                break
            changed = True
            cleanup_count += 1

            if animator and cleanup_count % 3 == 0:
                animator.add_step(
                    term["graph"],
                    title=f"Local Reductions (pass {cleanup_count})",
                    description="Applying theta, 2-cycle, triangle, degree-2, and loop reductions.",
                    operation="triangle"
                )

        # --- 2) TRY F-MOVE ---
        cycles = list_face_cycles(term["graph"])
        C = pick_smallest_interior_face_gt3(cycles) if cycles else None

        if not C or len(C) <= 3:
            if not changed:
                if animator:
                    animator.add_step(
                        term["graph"],
                        title="Final Reduced Graph",
                        description="All cycles reduced. Graph is now fully simplified!",
                        operation="triangle"
                    )
                return [term]
            continue

        new_term = f_move_recouple_term(term, C, i=0)

        if new_term is None:
            # A face > 3 exists but no F-move can be applied: stuck.
            if animator:
                animator.add_step(
                    term["graph"],
                    title="Reduction Complete (stuck)",
                    description=f"Cannot apply F-move on face {C}. Reduction terminates here.",
                    highlight_nodes=list(C),
                    operation="f-move"
                )
            return [term]

        if animator:
            animator.add_step(
                new_term["graph"],
                title=f"F-move Applied on {len(C)}-cycle",
                description=f"Applied F-move recoupling on face {C}. Inserted 6j symbol.",
                highlight_nodes=list(C),
                operation="f-move"
            )

        term = new_term

    # Safety cap: should not be reached for physically reasonable graphs.
    if animator:
        animator.add_step(
            term["graph"],
            title="Maximum Iterations Reached",
            description="Safety cap: Maximum iteration limit reached.",
            operation="triangle"
        )
    return [term]
