import networkx as nx

# -----------------------------------------------------------------------
# Triangular conditions
# -----------------------------------------------------------------------

# Spin networks require that the three spin labels at every trivalent vertex
# satisfy two conditions simultaneously:
#   1. Triangle inequality: each label ≤ sum of the other two.
#   2. Integer-sum rule: j1+j2+j3 must be an integer (ensures valid SU(2) coupling).
# This function checks both at once for a single set of three labels.
def vertex_satisfies_triangular_conditions(labels):
    a, b, c = labels
    if (a + b + c) % 1 != 0:  # integer-sum check
        return False
    return (
        abs(a - b) <= c <= a + b
        and abs(b - c) <= a <= b + c
        and abs(c - a) <= b <= c + a
    )

# Walk every trivalent vertex in the graph and raise ValueError if any vertex
# violates the triangular conditions. Non-numeric labels (e.g. symbolic F_k)
# are skipped because their validity is checked later during summation.
def check_triangular_condition(graph):
    for node in graph.nodes:
        neighbors = list(graph.neighbors(node))
        if len(neighbors) == 3:
            labels = []
            for neighbor in neighbors:
                edge_data = graph[node][neighbor]
                for key, data in edge_data.items():
                    label = data.get("label", None)
                    if isinstance(label, (int, float)):
                        labels.append(label)
            if len(labels) == 3 and not vertex_satisfies_triangular_conditions(labels):
                raise ValueError(f"Triangular condition not satisfied at node {node}")

# -----------------------------------------------------------------------
# Face cycles and topology
# -----------------------------------------------------------------------

# Spin network reduction works face by face. These helpers identify the faces
# (cycles) of the planar graph so that F-moves and triangle reductions can
# pick which face to attack next.

# Returns a list of node-lists, each representing one cycle.
# For planar graphs, uses the full planar embedding (enumerates all faces,
# which gives a richer set than a cycle basis alone).
# For non-planar graphs, falls back to minimum cycle basis — the reduction
# algorithm (F-moves, triangle reductions) only needs a list of cycles and
# is agnostic to how they were found.
def list_face_cycles(G):
    is_planar, certificate = nx.check_planarity(G)
    if not is_planar:
        return nx.minimum_cycle_basis(nx.Graph(G))

    faces = []
    visited = set()

    for u in certificate:
        for v in certificate[u]:
            if (u, v) in visited:
                continue
            face = list(certificate.traverse_face(u, v))
            faces.append(face)
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                visited.add((a, b))

    return faces

# Returns the second-largest face (skipping the outer/infinite face, which is
# the largest). Used historically; pick_smallest_interior_face_gt3 is preferred.
def pick_largest_interior_face(faces):
    if not faces:
        return None
    faces_sorted = sorted(faces, key=len, reverse=True)
    if len(faces_sorted) > 1:
        return faces_sorted[1]
    return faces_sorted[0]

# Returns the smallest interior face with more than 3 nodes.
# Triangles are handled by apply_triangle_reduction; this finds the next
# target for an F-move (which reduces a face's length by 1).
def pick_smallest_interior_face_gt3(cycles):
    cycles_gt3 = [c for c in cycles if len(c) > 3]
    if not cycles_gt3:
        return None
    return min(cycles_gt3, key=len)

# -----------------------------------------------------------------------
# Edge and neighbor helpers for trivalent graphs
# -----------------------------------------------------------------------

# In a trivalent (3-valent) graph every vertex has exactly 3 neighbours.
# When a vertex sits on a cycle, two of those neighbours are cycle-neighbours
# and one is external. This function returns that external neighbour.
def external_neighbor_in_trivalent(G, v, cycle_nodes):
    for nbr in G.neighbors(v):
        if nbr not in cycle_nodes:
            return nbr
    return None

# Returns the label on the first parallel edge found between u and v.
# In a MultiGraph there may be several; we only want the representative label.
def get_edge_label(G, u, v):
    if G.has_edge(u, v):
        for k, data in G[u][v].items():
            return data.get("label", None)
    return None

# Removes one edge instance between u and v (uses the first internal key).
# Used during graph reductions where a single edge must be surgically deleted
# without accidentally removing a parallel edge.
def remove_one_cycle_edge(G, u, v):
    if G.has_edge(u, v):
        k = next(iter(G[u][v].keys()))
        G.remove_edge(u, v, key=k)

# Adds a labeled edge between u and v. Thin wrapper kept for readability.
def add_edge_with_label(G, u, v, label):
    G.add_edge(u, v, label=label)

# In a 2-cycle (digon), vertex v has exactly 2 edges to 'other' and 1 external
# edge. Returns (external_neighbour, label_of_external_edge).
def single_external_neighbor_and_label(G, v, other):
    for nbr in G.neighbors(v):
        if nbr == other:
            continue
        lbl = get_edge_label(G, v, nbr)
        return nbr, lbl
    return None, None

# Returns all (label, neighbour) pairs for edges incident to v that do NOT go
# to 'other'. In a trivalent graph this gives the two non-pair edges, which
# are the two external edges of a digon vertex.
def incident_labels_excluding_pair(G, v, other):
    out = []
    for nbr in G.neighbors(v):
        if nbr == other:
            continue
        lbl = get_edge_label(G, v, nbr)
        out.append((lbl, nbr))
    return out

# Returns all labels on parallel edges between u and v.
# A digon has 2, a theta-graph vertex pair has 3.
def uv_parallel_labels(G, u, v):
    labels = []
    if G.has_edge(u, v):
        for k, data in G[u][v].items():
            labels.append(data.get("label", None))
    return labels

# -----------------------------------------------------------------------
# Numeric check and spin conversion
# -----------------------------------------------------------------------

# Edge labels can be numeric (1, 0.5, 1.5 …), symbolic strings ("F_1"),
# or sympy expressions (e.g. a + 2*b from parse_spin_label).
# This predicate is used throughout to decide whether to compute a numeric
# range or leave a label as a symbolic placeholder.
def is_numeric_label(x):
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, str):
        try:
            float(x)
            return True
        except ValueError:
            return False
    # Sympy numeric constants (e.g. Integer(2), Rational(1,2))
    try:
        import sympy
        if isinstance(x, sympy.Basic) and x.is_number:
            return True
    except ImportError:
        pass
    return False


# Parse a raw label string from the graph editor or a graphml file into the
# appropriate Python type:
#   - purely numeric string  → float   (e.g. "1.5"   → 1.5)
#   - single identifier      → str     (e.g. "F_1"   → "F_1",  backward-compatible)
#   - algebraic expression   → sympy   (e.g. "a+2b"  → a + 2*b)
#     Implicit multiplication is supported: "2b" means 2*b.
# Returns the original string unchanged on any parse failure.
def parse_spin_label(text: str):
    if not isinstance(text, str):
        return text
    text = text.strip()
    if not text:
        return text

    # 1. Numeric
    try:
        return float(text)
    except ValueError:
        pass

    # 2. Sympy with implicit multiplication ("2b" → 2*b)
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application,
        )
        expr = parse_expr(
            text,
            transformations=standard_transformations + (implicit_multiplication_application,),
            evaluate=True,
        )
        if expr.is_number:          # e.g. "3/2" → 1.5
            return float(expr)
        if expr.is_Symbol:          # simple identifier → keep as string
            return str(expr)
        return expr                 # compound expression → sympy Expr
    except Exception:
        pass

    # 3. Fallback: return as string
    return text

# Spins are stored as floats (0, 0.5, 1, 1.5 …). wigxjpf and many range
# calculations work in doubled integers (0, 1, 2, 3 …) to avoid fractions.
# This converts one spin value to its doubled integer.
def to_doubled(x):
    return int(round(2 * x))

# -----------------------------------------------------------------------
# Summation range computation for F-moves
# -----------------------------------------------------------------------

# When an F-move introduces a new intermediate spin F, its allowed range
# follows from two triangle inequalities simultaneously:
#   |b - d| ≤ F ≤ b + d   and   |a - c| ≤ F ≤ a + c
# So Fmin = max(|b-d|, |a-c|) and Fmax = min(b+d, a+c), both in doubled units.
# Returns None if the parity of the two bounds is inconsistent (no valid F).
# All arguments must be numeric; use f_range_with_symbolic for mixed cases.
def f_range_symbolic(a, b, d, c):
    if not all(map(is_numeric_label, [a, b, d, c])):
        return None
    A, B, D, C = map(to_doubled, (a, b, d, c))
    Fmin = max(abs(B - D), abs(A - C))
    Fmax = min(B + D, A + C)
    parity_ac = (A + C) % 2
    parity_bd = (B + D) % 2
    if parity_ac != parity_bd:
        return None
    return {"Fmin": Fmin, "Fmax": Fmax, "parity": parity_ac}


# Extended version of f_range_symbolic that also handles symbolic labels
# (e.g. a previous F-variable). When labels are numeric, delegates to
# f_range_symbolic for exact bounds. When some labels are symbolic, uses
# their known ranges (from known_ranges dict) to compute tighter numeric
# bounds than the conservative default [0, 40].
#
# Returns a dict with:
#   Fmin, Fmax  – tight bounds in doubled units
#   parity      – expected parity (0 if symbolic; exact if numeric)
#   symbolic    – True if any label was symbolic
#   depends_on  – list of symbolic variable names that affect this range
def f_range_with_symbolic(a, b, d, c, known_ranges=None):
    if known_ranges is None:
        known_ranges = {}

    all_numeric = all(map(is_numeric_label, [a, b, d, c]))

    if all_numeric:
        rng = f_range_symbolic(a, b, d, c)
        if rng is not None:
            rng["symbolic"] = False
            return rng
        else:
            return None

    symbolic_labels = [lbl for lbl in [a, b, d, c] if not is_numeric_label(lbl)]

    def get_value_range(x):
        # Return (min, max) in doubled units for a label.
        # For unknowns, fall back to a conservative [0, 40] (j up to 20).
        if is_numeric_label(x):
            doubled = to_doubled(x)
            return (doubled, doubled)
        elif x in known_ranges:
            return known_ranges[x]
        else:
            return (0, 40)

    a_min, a_max = get_value_range(a)
    b_min, b_max = get_value_range(b)
    c_min, c_max = get_value_range(c)
    d_min, d_max = get_value_range(d)

    bd_diff_min = max(0, abs(b_min - d_max) if b_min >= d_max else 0,
                         abs(b_max - d_min) if b_max <= d_min else 0)
    bd_diff_max = max(abs(b_min - d_min), abs(b_max - d_max),
                      abs(b_min - d_max), abs(b_max - d_min))

    ac_diff_min = max(0, abs(a_min - c_max) if a_min >= c_max else 0,
                         abs(a_max - c_min) if a_max <= c_min else 0)
    ac_diff_max = max(abs(a_min - c_min), abs(a_max - c_max),
                      abs(a_min - c_max), abs(a_max - c_min))

    bd_sum_max = b_max + d_max
    ac_sum_max = a_max + c_max

    fmin = max(bd_diff_min, ac_diff_min)
    fmax = min(bd_sum_max, ac_sum_max)

    if fmin > fmax:
        return None

    def build_expr(op, x, y):
        x_str = str(x)
        y_str = str(y)
        if op == 'abs_diff':
            return f"abs({x_str} - ({y_str}))"
        elif op == 'sum':
            return f"{x_str} + {y_str}"
        elif op == 'max':
            return f"max({x_str}, {y_str})"
        elif op == 'min':
            return f"min({x_str}, {y_str})"
        else:
            return f"{op}({x_str}, {y_str})"

    bd_diff = build_expr('abs_diff', b, d) if not is_numeric_label(b) or not is_numeric_label(d) else str(abs(to_doubled(b) - to_doubled(d)))
    ac_diff = build_expr('abs_diff', a, c) if not is_numeric_label(a) or not is_numeric_label(c) else str(abs(to_doubled(a) - to_doubled(c)))
    fmin_expr = build_expr('max', bd_diff, ac_diff)

    bd_sum = build_expr('sum', b, d) if not is_numeric_label(b) or not is_numeric_label(d) else str(to_doubled(b) + to_doubled(d))
    ac_sum = build_expr('sum', a, c) if not is_numeric_label(a) or not is_numeric_label(c) else str(to_doubled(a) + to_doubled(c))
    fmax_expr = build_expr('min', bd_sum, ac_sum)

    return {
        "Fmin": fmin,
        "Fmax": fmax,
        "parity": 0,
        "symbolic": True,
        "symbolic_Fmin": fmin_expr,
        "symbolic_Fmax": fmax_expr,
        "depends_on": symbolic_labels
    }


# When F-moves are nested (one F-variable appears in the range of another),
# we can tighten bounds by iterating outermost-first and recording each
# variable's numeric range as it is resolved. Returns {var: (Fmin, Fmax)}.
def compute_nested_ranges(summation_variables, all_ranges_info):
    known_ranges = {}

    for var in summation_variables:
        if var not in all_ranges_info:
            known_ranges[var] = (0, 40)
            continue

        range_info = all_ranges_info[var]

        if not range_info.get("symbolic", False):
            known_ranges[var] = (range_info["Fmin"], range_info["Fmax"])
        else:
            known_ranges[var] = (range_info["Fmin"], range_info["Fmax"])

    return known_ranges
