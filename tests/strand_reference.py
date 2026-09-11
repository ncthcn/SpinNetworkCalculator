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

"""
Independent spin-network evaluator, built from the strand definition.

This deliberately shares NO code with src/: no 6j symbols, no theta, no
graph reduction. It implements the combinatorial definition directly --

    expand every spin-j edge into 2j parallel strands, antisymmetrise along
    each edge, route the strands through each trivalent vertex, count the
    closed loops of each resulting configuration and weight it by (-2)^loops

-- so that anything it agrees with in src/ is corroborated by a genuinely
separate calculation. It is exponential in the total number of strands and is
only usable on small graphs, which is exactly what a reference implementation
needs to be.

Conventions
-----------
At a trivalent vertex joining edges a, b, c the strand counts routed between
each pair are

    n_ab = j_a + j_b - j_c,   n_bc = j_b + j_c - j_a,   n_ca = j_c + j_a - j_b

each a non-negative integer for an admissible vertex, with
n_ab + n_ca = 2*j_a and so on.

Two normalisations are offered because conventions differ on whether the
antisymmetriser carries a 1/(2j)! :

    evaluate(g, normalise=False)  ->  sum over permutations, unnormalised
    evaluate(g, normalise=True)   ->  divided by prod (2j_e)!
"""

from __future__ import annotations

import itertools
from math import factorial


def _sign(perm):
    """Parity of a permutation given as a tuple of images of 0..n-1."""
    seen = [False] * len(perm)
    parity = 1
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = perm[node]
            length += 1
        if length % 2 == 0:
            parity = -parity
    return parity


def _ekey(u, v, k):
    """
    Canonical identifier for a MultiGraph edge.

    NetworkX reports an edge as (u, v, k) or (v, u, k) depending on which
    endpoint you queried, so the endpoints must be ordered to give both ends
    the same key.
    """
    a, b = str(u), str(v)
    return (a, b, k) if a <= b else (b, a, k)


def vertex_phase(a, b, c):
    """
    Sign picked up by swapping any two edges at a vertex carrying (a, b, c).

        (b, a, c) = (-1)**(a+b+c+4*(a*b + b*c + a*c)) * (a, b, c)

    The exponent is symmetric in a, b, c, so every pairwise transposition at
    the vertex carries the same factor. The 4*(...) makes the exponent an
    integer even for half-integer spins.
    """
    exponent = (a + b + c) + 4 * (a * b + b * c + a * c)
    return (-1.0) ** int(round(exponent))


def _vertex_matching(graph, orientation=None):
    """
    Pair up strand ports at every trivalent vertex.

    A port is the triple (edge_key, node, index) where edge_key identifies one
    edge of the MultiGraph and index runs over that edge's 2j strands at that
    node's end.

    Returns a dict mapping each port to the port it is joined to.
    """
    match = {}
    for node in graph.nodes():
        incident = [(u, v, k, d) for u, v, k, d in graph.edges(node, keys=True, data=True)]
        if orientation is not None and node in orientation:
            # Re-order this vertex's incident edges into the caller's cyclic
            # order, so the sign convention is explicit rather than whatever
            # networkx happened to return.
            wanted = list(orientation[node])
            by_key = {_ekey(u, v, k): (u, v, k, d) for u, v, k, d in incident}
            if sorted(map(str, wanted)) == sorted(map(str, by_key)):
                incident = [by_key[k] for k in wanted]
        if len(incident) != 3:
            raise ValueError(
                f"node {node!r} has degree {len(incident)}; the strand "
                "reference implementation handles trivalent vertices only"
            )
        # A self-loop would appear twice; not supported here.
        labels = [float(d["label"]) for _u, _v, _k, d in incident]
        ja, jb, jc = labels
        n_ab = ja + jb - jc
        n_bc = jb + jc - ja
        n_ca = jc + ja - jb
        for n in (n_ab, n_bc, n_ca):
            if n < -1e-9 or abs(n - round(n)) > 1e-9:
                raise ValueError(
                    f"vertex {node!r} with labels {labels} is inadmissible "
                    f"(pair counts {n_ab}, {n_bc}, {n_ca})"
                )
        n_ab, n_bc, n_ca = int(round(n_ab)), int(round(n_bc)), int(round(n_ca))

        keys = [_ekey(u, v, k) for u, v, k, _d in incident]
        a, b, c = keys

        def port(edge, idx):
            return (edge, node, idx)

        # a[0:n_ab]  <->  b[0:n_ab]
        for i in range(n_ab):
            match[port(a, i)] = port(b, i)
            match[port(b, i)] = port(a, i)
        # b[n_ab:n_ab+n_bc]  <->  c[0:n_bc]
        for i in range(n_bc):
            match[port(b, n_ab + i)] = port(c, i)
            match[port(c, i)] = port(b, n_ab + i)
        # c[n_bc:n_bc+n_ca]  <->  a[n_ab:n_ab+n_ca]
        for i in range(n_ca):
            match[port(c, n_bc + i)] = port(a, n_ab + i)
            match[port(a, n_ab + i)] = port(c, n_bc + i)
    return match


def evaluate(graph, normalise=False, loop_value=-2.0, orientation=None):
    """
    Evaluate a closed trivalent spin network by strand expansion.

    Parameters
    ----------
    graph : nx.MultiGraph
        Closed (no degree-1 nodes), every node trivalent, every edge carrying
        a numeric 'label' (the spin j).
    normalise : bool
        Divide by prod (2j_e)! -- i.e. use normalised antisymmetrisers.
    loop_value : float
        Value assigned to each closed strand loop. -2 is the SU(2) convention.

    Returns
    -------
    float
    """
    edges = [(u, v, k, float(d["label"])) for u, v, k, d in graph.edges(keys=True, data=True)]
    for node, deg in graph.degree():
        if deg != 3:
            raise ValueError(f"node {node!r} has degree {deg}; expected 3")

    vmatch = _vertex_matching(graph, orientation)

    widths = [int(round(2 * j)) for _u, _v, _k, j in edges]
    perms_per_edge = [list(itertools.permutations(range(w))) for w in widths]

    total = 0.0
    for choice in itertools.product(*perms_per_edge):
        # Build the edge matching for this term.
        ematch = {}
        sign = 1
        for (u, v, k, _j), width, perm in zip(edges, widths, choice):
            sign *= _sign(perm)
            key = _ekey(u, v, k)
            for i in range(width):
                p_u = (key, u, i)
                p_v = (key, v, perm[i])
                ematch[p_u] = p_v
                ematch[p_v] = p_u

        # Count cycles in the union of the two perfect matchings.
        unvisited = set(ematch)
        loops = 0
        while unvisited:
            start = next(iter(unvisited))
            node = start
            while True:
                unvisited.discard(node)
                node = ematch[node]
                unvisited.discard(node)
                node = vmatch[node]
                if node == start:
                    break
            loops += 1

        total += sign * (loop_value ** loops)

    if normalise:
        for w in widths:
            total /= factorial(w)
    return total
