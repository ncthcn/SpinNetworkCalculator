import networkx as nx

# -------------------------------
# --- Triangular condition
# -------------------------------

def vertex_satisfies_triangular_conditions(labels):
    a, b, c = labels
    if (a + b + c) % 1 != 0:
        return False
    return (
        abs(a - b) <= c <= a + b
        and abs(b - c) <= a <= b + c
        and abs(c - a) <= b <= c + a
    )

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
            
# -----------------------
# Helpers: cycles and topology
# -----------------------

def list_face_cycles(G):
    """
    Return all face cycles of a planar graph G using a planar embedding.
    Each face is a list of nodes forming a simple cycle.
    """
    is_planar, embedding = nx.check_planarity(G)
    if not is_planar:
        raise ValueError("Graph must be planar")

    faces = []
    visited = set()

    for u in embedding:
        for v in embedding[u]:
            if (u, v) in visited:
                continue

            face = list(embedding.traverse_face(u, v))
            faces.append(face)

            # mark directed edges of this face as visited
            for i in range(len(face)):
                a = face[i]
                b = face[(i + 1) % len(face)]
                visited.add((a, b))

    return faces

def pick_largest_interior_face(faces):
    if not faces:
        return None

    faces_sorted = sorted(faces, key=len, reverse=True)

    # If more than one face, skip the outer face (largest)
    if len(faces_sorted) > 1:
        return faces_sorted[1]

    return faces_sorted[0]

def pick_smallest_interior_face_gt3(cycles):
    """
    Return the smallest interior cycle with length > 3.
    """
    cycles_gt3 = [c for c in cycles if len(c) > 3]
    if not cycles_gt3:
        return None
    return min(cycles_gt3, key=len)


def external_neighbor_in_trivalent(G, v, cycle_nodes):
    """
    Find the non-cycle neighbor of v (unique in 3-valent graphs).
    Returns None if not found.
    """
    for nbr in G.neighbors(v):
        if nbr not in cycle_nodes:
            return nbr
    return None

def get_edge_label(G, u, v):
    """Return the label of the first found edge between u and v (MultiGraph)."""
    if G.has_edge(u, v):
        for k, data in G[u][v].items():
            return data.get("label", None)
    return None

def remove_one_cycle_edge(G, u, v):
    """Remove one edge instance between u and v (first key)."""
    if G.has_edge(u, v):
        k = next(iter(G[u][v].keys()))
        G.remove_edge(u, v, key=k)

def add_edge_with_label(G, u, v, label):
    """Add a labeled edge between u and v."""
    G.add_edge(u, v, label=label)

def single_external_neighbor_and_label(G, v, other):
    """
    Return (neighbor, label) for the unique neighbor of v that is not 'other',
    in a 3-valent setting where v has two edges to 'other' and one external edge.
    Returns (None, None) if not found.
    """
    for nbr in G.neighbors(v):
        if nbr == other:
            continue
        lbl = get_edge_label(G, v, nbr)
        return nbr, lbl
    return None, None

def incident_labels_excluding_pair(G, v, other):
    """
    Return list of (label, neighbor) for edges incident to v excluding any edges to 'other'.
    Assumes 3-valent; should return two entries.
    """
    out = []
    for nbr in G.neighbors(v):
        if nbr == other:
            continue
        lbl = get_edge_label(G, v, nbr)
        out.append((lbl, nbr))
    return out

def uv_parallel_labels(G, u, v):
    """Return list of labels on parallel edges between u and v."""
    labels = []
    if G.has_edge(u, v):
        for k, data in G[u][v].items():
            labels.append(data.get("label", None))
    return labels

# -----------------------
# Helpers: numeric check and symbolic range
# -----------------------

def is_numeric_label(x):
    return isinstance(x, (int, float))

def to_doubled(x):
    """Convert half-integer / integer float to doubled integer."""
    return int(round(2 * x))

def f_range_symbolic(a, b, d, c):
    """
    Compute compact symbolic range for the doubled summation index F
    given (a, b, d, c) in half-integers.
    Returns dict {Fmin, Fmax, parity} or None if inconsistent parity or non-numeric.
    """
    if not all(map(is_numeric_label, [a, b, d, c])):
        return None
    A, B, D, C = map(to_doubled, (a, b, d, c))
    Fmin = max(abs(B - D), abs(A - C))
    Fmax = min(B + D, A + C)
    parity_ac = (A + C) % 2
    parity_bd = (B + D) % 2
    if parity_ac != parity_bd:
        return None  # inconsistent: no allowed f values
    return {"Fmin": Fmin, "Fmax": Fmax, "parity": parity_ac}


def f_range_with_symbolic(a, b, d, c):
    """
    Compute range for F summation, handling both numeric and symbolic edge labels.

    When all labels are numeric: returns exact {Fmin, Fmax, parity}
    When some labels are symbolic (e.g., F_1): returns symbolic expressions

    Parameters:
    -----------
    a, b, d, c : edge labels (numeric or symbolic strings like "F_1")

    Returns:
    --------
    dict with keys:
        - "Fmin": int or symbolic expression (string)
        - "Fmax": int or symbolic expression (string)
        - "parity": int or symbolic expression (string)
        - "symbolic": bool (True if any label is symbolic)

    For symbolic ranges, we use conservative bounds that will be resolved
    during numerical evaluation when the F-variables are substituted.
    """
    # Check if all labels are numeric
    all_numeric = all(map(is_numeric_label, [a, b, d, c]))

    if all_numeric:
        # Use existing function for pure numeric case
        rng = f_range_symbolic(a, b, d, c)
        if rng is not None:
            rng["symbolic"] = False
            return rng
        else:
            return None

    # At least one label is symbolic (e.g., F_1, F_2, ...)
    # We need to create a symbolic range expression

    # Collect numeric and symbolic labels
    labels = {'a': a, 'b': b, 'd': d, 'c': c}
    symbolic_labels = [lbl for lbl in [a, b, d, c] if not is_numeric_label(lbl)]

    if not symbolic_labels:
        # Should not reach here, but fallback
        return None

    # For symbolic ranges, we use expressions that will be evaluated later
    # Triangle inequality: |b-d| <= F <= b+d  AND  |a-c| <= F <= a+c
    # So: Fmin = max(|b-d|, |a-c|)  and  Fmax = min(b+d, a+c)

    # Build symbolic expressions as strings
    def build_expr(op, x, y):
        """Build expression string like 'max(|b-d|, |a-c|)'"""
        x_str = str(x) if is_numeric_label(x) else str(x)
        y_str = str(y) if is_numeric_label(y) else str(y)

        if op == 'abs_diff':
            return f"abs({x_str} - {y_str})"
        elif op == 'sum':
            return f"{x_str} + {y_str}"
        elif op == 'max':
            return f"max({x_str}, {y_str})"
        elif op == 'min':
            return f"min({x_str}, {y_str})"
        else:
            return f"{op}({x_str}, {y_str})"

    # Fmin = max(|b-d|, |a-c|)
    bd_diff = build_expr('abs_diff', b, d) if not is_numeric_label(b) or not is_numeric_label(d) else str(abs(to_doubled(b) - to_doubled(d)))
    ac_diff = build_expr('abs_diff', a, c) if not is_numeric_label(a) or not is_numeric_label(c) else str(abs(to_doubled(a) - to_doubled(c)))

    fmin_expr = build_expr('max', bd_diff, ac_diff)

    # Fmax = min(b+d, a+c)
    bd_sum = build_expr('sum', b, d) if not is_numeric_label(b) or not is_numeric_label(d) else str(to_doubled(b) + to_doubled(d))
    ac_sum = build_expr('sum', a, c) if not is_numeric_label(a) or not is_numeric_label(c) else str(to_doubled(a) + to_doubled(c))

    fmax_expr = build_expr('min', bd_sum, ac_sum)

    # For now, we'll use a conservative default range
    # The actual range will be computed during numerical evaluation
    # Use a reasonable default: 0 to max_expected_j (we'll use 20 as conservative upper bound)

    return {
        "Fmin": 0,  # Conservative lower bound (will be tightened during evaluation)
        "Fmax": 40,  # Conservative upper bound (2*j for j=20, will be tightened)
        "parity": 0,  # Will be checked during evaluation
        "symbolic": True,
        "symbolic_Fmin": fmin_expr,
        "symbolic_Fmax": fmax_expr,
        "depends_on": symbolic_labels
    }
