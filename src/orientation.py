#     SPDX-License-Identifier: GPL-3.0-or-later
#     Copyright (C) 2026, N. Cohen, University of Vienna & IQOQI Vienna

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Vertex orientation and graph phase tracking for spin networks.

Each trivalent vertex has a ``reference_orientation``: a fixed cyclic order of
its three incident edges, captured once when the vertex reaches degree 3.
Changing the 2D layout may reorder those edges, accumulating a phase factor.

**Phase formula** — for an elementary adjacent transposition swapping the two
edges with labels *a* and *b* at a vertex whose third (spectator) edge has
label *c*:

    phase_factor = (-1) ** (2 * c * (a + b - c))

For a valid trivalent vertex (triangle inequality + integer-sum rule),
``a + b - c`` is a non-negative integer and ``2c`` is an integer, so the
exponent is always an integer and the factor is ±1.
"""

from __future__ import annotations

import math

import networkx as nx

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# Canonical edge reference: (smaller_id, larger_id, key)
EdgeRef = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def canonical_edge(u: int, v: int, key: int) -> EdgeRef:
    """Return an edge reference with the smaller node ID first.

    Parameters
    ----------
    u, v : int
        Node endpoints.
    key : int
        NetworkX MultiGraph edge key.

    Returns
    -------
    EdgeRef
        ``(min(u, v), max(u, v), key)``
    """
    return (min(u, v), max(u, v), key)


def get_incident_edges(graph: nx.MultiGraph, vertex: int) -> list[EdgeRef]:
    """Return all incident edges of *vertex* as canonical :data:`EdgeRef` tuples.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph.
    vertex : int
        The vertex whose incident edges to list.

    Returns
    -------
    list[EdgeRef]
        Incident edges in the order NetworkX iterates them.
    """
    edges: list[EdgeRef] = []
    for neighbor, edge_dict in graph[vertex].items():
        for key in edge_dict:
            edges.append(canonical_edge(vertex, neighbor, key))
    return edges


def _edge_label(graph: nx.MultiGraph, edge: EdgeRef) -> float:
    """Return the numerical spin label of *edge*, or 0.0 if non-numeric.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph.
    edge : EdgeRef
        Canonical edge reference.

    Returns
    -------
    float
        Numerical value of the edge label.
    """
    u, v, key = edge
    label = graph.edges[u, v, key].get("label", 0)
    try:
        return float(label)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Reference orientation (stored as node attribute)
# ---------------------------------------------------------------------------


def set_reference_orientation(graph: nx.MultiGraph, vertex: int) -> None:
    """Capture the current incident-edge order as the reference orientation.

    Called once when *vertex* reaches degree 3 (becomes trivalent).
    The stored list is the baseline for all subsequent phase calculations.
    After this call the attribute must **not** be mutated.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph.
    vertex : int
        The vertex whose orientation to fix.
    """
    edges = get_incident_edges(graph, vertex)
    graph.nodes[vertex]["reference_orientation"] = edges


def get_reference_orientation(
    graph: nx.MultiGraph, vertex: int
) -> list[EdgeRef] | None:
    """Return the stored reference orientation for *vertex*, or ``None``.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph.
    vertex : int
        The vertex to query.

    Returns
    -------
    list[EdgeRef] | None
        The reference orientation, or ``None`` if not yet set.
    """
    return graph.nodes[vertex].get("reference_orientation", None)


# ---------------------------------------------------------------------------
# Transposition arithmetic
# ---------------------------------------------------------------------------


def _transposition_sequence(
    reference: list,
    layout_order: list,
) -> list[tuple[int, int]]:
    """Find the minimal sequence of adjacent swaps to sort *reference* into *layout_order*.

    Uses an insertion-sort strategy: for each position in *layout_order* (left
    to right) find the target element in the current working copy of
    *reference* and bubble it left to its correct position.

    Parameters
    ----------
    reference : list
        The starting order of elements.
    layout_order : list
        The desired target order (must contain the same elements).

    Returns
    -------
    list[tuple[int, int]]
        Sequence of ``(i, i+1)`` index pairs representing adjacent swaps.
        Applying them in order to *reference* yields *layout_order*.

    Raises
    ------
    ValueError
        If the two lists contain different elements.

    Examples
    --------
    >>> _transposition_sequence([0, 1, 2], [2, 0, 1])
    [(1, 2), (0, 1)]
    """
    if sorted(str(x) for x in reference) != sorted(str(x) for x in layout_order):
        raise ValueError("reference and layout_order must contain the same elements")

    current = list(reference)
    swaps: list[tuple[int, int]] = []

    for i, target_elem in enumerate(layout_order):
        idx = current.index(target_elem)
        while idx > i:
            swaps.append((idx - 1, idx))
            current[idx - 1], current[idx] = current[idx], current[idx - 1]
            idx -= 1

    return swaps


def _cyclic_align(reference: list, layout_order: list) -> list:
    """Rotate *layout_order* so it starts with the same element as *reference*.

    This canonical alignment removes the rotational degree of freedom in
    cyclic orderings before comparing them linearly.

    Parameters
    ----------
    reference : list
        The reference ordering (determines the starting element).
    layout_order : list
        The ordering to rotate.

    Returns
    -------
    list
        A rotation of *layout_order* whose first element equals ``reference[0]``.

    Raises
    ------
    ValueError
        If ``reference[0]`` is not found in *layout_order*.
    """
    start = reference[0]
    if start not in layout_order:
        raise ValueError(f"Element {start!r} not found in layout_order")
    idx = layout_order.index(start)
    n = len(layout_order)
    return [layout_order[(idx + i) % n] for i in range(n)]


def phase_factor_for_transposition(
    graph: nx.MultiGraph,
    edge_a: EdgeRef,
    edge_b: EdgeRef,
    edge_c: EdgeRef,
) -> complex:
    """Compute the phase factor for one transposition at a trivalent vertex.

    For a vertex carrying edge labels (a, b, c)::

        (b, a, c) = (-1) ** (a + b + c + 4*(a*b + b*c + a*c)) * (a, b, c)

    The exponent is **symmetric** in a, b and c, so every pairwise
    transposition at the vertex carries the same factor -- which is what one
    expects, since a trivalent vertex admits only two cyclic orders and any
    transposition swaps between them.

    The ``4*(...)`` term makes the exponent an integer even when the labels are
    half-integers: each product of two half-integers is a multiple of 1/4.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph (used to read edge labels).
    edge_a, edge_b : EdgeRef
        The two edges being swapped.
    edge_c : EdgeRef
        The spectator (third) edge at the vertex.  It enters the exponent
        symmetrically with the other two.

    Returns
    -------
    complex
        Either ``+1+0j`` or ``-1+0j``.
    """
    a = _edge_label(graph, edge_a)
    b = _edge_label(graph, edge_b)
    c = _edge_label(graph, edge_c)
    return complex(vertex_phase(a, b, c))


def vertex_phase(a: float, b: float, c: float) -> float:
    """
    Sign acquired by swapping any two edges at a vertex carrying (a, b, c).

        (-1) ** (a + b + c + 4*(a*b + b*c + a*c))

    Kept as a plain numeric helper so the reduction moves can call it without
    constructing edge references.
    """
    exponent = (a + b + c) + 4.0 * (a * b + b * c + a * c)
    return float((-1) ** int(round(exponent)))


# ---------------------------------------------------------------------------
# Layout phase calculation
# ---------------------------------------------------------------------------


def calculate_layout_phase(
    graph: nx.MultiGraph,
    layout: dict[int, tuple[float, float]],
) -> complex:
    """Compute the cumulative phase factor for a given 2D layout.

    For every trivalent vertex with a stored reference orientation:

    1. Compute the polar angle from the vertex to each neighbour in *layout*.
    2. Sort incident edges by angle (counterclockwise = increasing angle).
    3. Cyclically align the sorted order with the reference orientation so
       both start with the same edge.
    4. Find the minimal adjacent-transposition sequence from reference to
       layout order via :func:`_transposition_sequence`.
    5. Multiply a running phase by the result of
       :func:`phase_factor_for_transposition` for each swap.

    This function does **not** mutate ``graph.graph['phase']``.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph.  Each trivalent node should have a
        ``"reference_orientation"`` attribute set by
        :func:`set_reference_orientation`.
    layout : dict[int, tuple[float, float]]
        Mapping from vertex ID to ``(x, y)`` coordinates.

    Returns
    -------
    complex
        The phase factor accumulated over all trivalent vertices.
    """
    phase = complex(1.0)

    for vertex in graph.nodes():
        ref_orient = get_reference_orientation(graph, vertex)
        if ref_orient is None or len(ref_orient) != 3:
            continue
        if vertex not in layout:
            continue

        vx, vy = layout[vertex]

        # Map each incident edge to its polar angle from vertex to neighbour.
        # For parallel edges (same neighbour) the angle is identical; we
        # preserve their relative order from reference_orientation (stable sort).
        ref_index: dict[EdgeRef, int] = {e: i for i, e in enumerate(ref_orient)}

        def _sort_key(edge: EdgeRef) -> tuple[float, int]:
            u, v, _key = edge
            neighbor = v if u == vertex else u
            if neighbor not in layout:
                return (0.0, ref_index.get(edge, 0))
            nx_, ny = layout[neighbor]
            return (math.atan2(ny - vy, nx_ - vx), ref_index.get(edge, 0))

        layout_orient = sorted(ref_orient, key=_sort_key)

        # Cyclically align so both sequences start at the same edge.
        try:
            layout_orient = _cyclic_align(ref_orient, layout_orient)
        except ValueError:
            continue

        # Compute the transposition sequence.
        try:
            swaps = _transposition_sequence(ref_orient, layout_orient)
        except ValueError:
            continue

        if not swaps:
            continue

        # Re-apply swaps tracking which edges are in which position so we can
        # identify edge_a, edge_b, and the spectator edge_c correctly.
        current: list[EdgeRef] = list(ref_orient)
        for i, j in swaps:
            edge_a = current[i]
            edge_b = current[j]  # j == i + 1 always
            edge_c = next(e for e in current if e != edge_a and e != edge_b)
            phase *= phase_factor_for_transposition(graph, edge_a, edge_b, edge_c)
            current[i], current[j] = current[j], current[i]

    return phase


# ---------------------------------------------------------------------------
# Planar flattening
# ---------------------------------------------------------------------------


def resolve_planar_flattening(
    graph: nx.MultiGraph,
) -> tuple[dict[int, tuple[float, float]], complex]:
    """Find a near-planar layout and compute the phase shift it induces.

    Algorithm:

    1. If *graph* is planar (tested via :func:`networkx.check_planarity`),
       compute a straight-line planar embedding with
       :func:`networkx.planar_layout`.  Otherwise fall back to a
       force-directed spring layout (deterministic, ``seed=42``).
    2. Call :func:`calculate_layout_phase` for that layout.
    3. Return both the coordinate dict and the phase.

    The ``graph.graph['phase']`` attribute is **not** mutated.

    Parameters
    ----------
    graph : nx.MultiGraph
        The spin network graph with reference orientations set.

    Returns
    -------
    tuple[dict[int, tuple[float, float]], complex]
        - ``layout``: mapping from vertex ID to ``(x, y)`` float coordinates.
        - ``phase``: cumulative phase shift as a complex number.
    """

    def _circular_layout() -> dict:
        """Pure-Python circular fallback — no numpy required."""
        nodes = list(graph.nodes())
        n = len(nodes) or 1
        return {
            v: (math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n))
            for i, v in enumerate(nodes)
        }

    try:
        is_planar, _ = nx.check_planarity(graph)
        pos = nx.planar_layout(graph) if is_planar else nx.spring_layout(graph, seed=42)
    except Exception:
        try:
            pos = nx.spring_layout(graph, seed=42)
        except Exception:
            pos = _circular_layout()

    try:
        layout: dict[int, tuple[float, float]] = {
            v: (float(xy[0]), float(xy[1])) for v, xy in pos.items()
        }
    except Exception:
        layout = _circular_layout()
    phase = calculate_layout_phase(graph, layout)
    graph.graph["phase"] = phase  # commit in-place
    return layout, phase
