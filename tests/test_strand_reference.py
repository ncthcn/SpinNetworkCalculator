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
Validate src/ against an independent evaluator built from the strand definition.

tests/strand_reference.py implements the definition quoted in README.md --
expand each spin-j edge into 2j strands, antisymmetrise, count closed loops,
weight each by (-2) -- sharing no code with src/. It is exponential in the
total strand count, so it is only usable on small networks; that is exactly
what makes it a trustworthy oracle.

WHY THIS FILE MATTERS
---------------------
Every other test in the suite checks src/ against itself or against sympy's 6j
symbols. Neither catches an error in the *graph reduction*, because both sides
would be wrong together. This file closes that gap: it compares the full
pipeline (glue -> F-moves -> triangle reductions -> canonicalise -> evaluate)
against a calculation that knows nothing about 6j symbols.

Doing so immediately exposed a real defect; see TestKnownDefect at the end.
"""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from math import factorial

from src.api import Graph
from tests.strand_reference import evaluate


# ---------------------------------------------------------------------------
# Small network builders
# ---------------------------------------------------------------------------

def _pos(g):
    for n in g.nodes:
        g.nodes[n]["pos"] = (0.0, 0.0)
    return g


def theta_net(labels=(1.0, 1.0, 1.0)):
    """Two vertices joined by three edges."""
    g = nx.MultiGraph()
    for label in labels:
        g.add_edge("u", "v", label=label)
    return _pos(g)


def tetrahedron(j=1.0):
    g = nx.MultiGraph()
    n = ["t0", "t1", "t2", "t3"]
    for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        g.add_edge(n[a], n[b], label=j)
    return _pos(g)


def cube(j=1.0):
    """Two squares joined by four rungs -- the glued form of ring_with_legs(4)."""
    g = nx.MultiGraph()
    a = [f"a{i}" for i in range(4)]
    b = [f"b{i}" for i in range(4)]
    for i in range(4):
        g.add_edge(a[i], a[(i + 1) % 4], label=j)
        g.add_edge(b[i], b[(i + 1) % 4], label=j)
        g.add_edge(a[i], b[i], label=j)
    return _pos(g)


def ring_with_legs(n, j=1.0):
    """An n-cycle of trivalent vertices, each carrying one open leg."""
    g = nx.MultiGraph()
    ring = [f"n{i}" for i in range(n)]
    for i, node in enumerate(ring):
        g.add_edge(node, ring[(i + 1) % n], label=j)
        g.add_edge(node, f"L{i}", label=j)
    return _pos(g)


# ---------------------------------------------------------------------------
# 1. The oracle itself, checked against the closed-form theta net
# ---------------------------------------------------------------------------

class TestOracleIsItselfCorrect:
    """
    Before trusting the oracle, check it against the Kauffman-Lins closed form
    for the theta net, which is independent of both it and src/.
    """

    @staticmethod
    def kauffman_lins_theta(a, b, c):
        m = int(round(a + b - c))
        n = int(round(b + c - a))
        p = int(round(c + a - b))
        return (
            factorial(int(round(a + b + c)) + 1)
            * factorial(m) * factorial(n) * factorial(p)
            / (factorial(int(round(2 * a)))
               * factorial(int(round(2 * b)))
               * factorial(int(round(2 * c))))
        )

    @pytest.mark.parametrize("labels", [
        (1.0, 1.0, 1.0), (1.0, 0.5, 0.5), (1.0, 1.0, 2.0),
        (1.5, 1.5, 1.0), (2.0, 2.0, 2.0), (0.5, 1.0, 1.5),
    ])
    def test_matches_kauffman_lins_theta(self, labels):
        got = evaluate(theta_net(labels), normalise=True)
        expected = self.kauffman_lins_theta(*labels)
        assert abs(got) == pytest.approx(expected, rel=1e-9), (
            f"strand expansion gives |{got}| for theta{labels}, "
            f"Kauffman-Lins closed form gives {expected}"
        )

    def test_inadmissible_vertex_is_rejected(self):
        with pytest.raises(ValueError, match="inadmissible"):
            evaluate(theta_net((1.0, 1.0, 5.0)))


# ---------------------------------------------------------------------------
# 2. src/ against the oracle
# ---------------------------------------------------------------------------

class TestPipelineAgainstOracle:
    """
    ||G|| for a CLOSED network G is the value of two disjoint copies of G,
    because gluing has no open ends to weld -- hence oracle(G)**2.
    """

    @pytest.mark.parametrize("name,builder,expected_value", [
        ("theta j=1",       lambda: theta_net((1.0, 1.0, 1.0)), -24.0),
        ("tetrahedron j=1", lambda: tetrahedron(1.0),            96.0),
        ("cube j=1",        lambda: cube(1.0),                 6144.0),
    ])
    def test_closed_network_norm_is_the_oracle_squared(self, name, builder, expected_value):
        reference = evaluate(builder())
        if expected_value is not None:
            assert reference == pytest.approx(expected_value), (
                f"{name}: oracle drifted from its documented value"
            )
        got = Graph(builder()).evaluate_symbolic().evaluate_numeric()
        assert got == pytest.approx(abs(reference ** 2), rel=1e-9), (
            f"{name}: pipeline gives {got!r}, oracle squared is {reference ** 2!r}"
        )

    def test_open_network_norm_equals_its_glued_value(self):
        """
        Gluing a 3-ring's open legs to the mirror copy gives a closed network;
        the norm must equal that network's value.
        """
        from src.gluer import glue_open_edges
        g = ring_with_legs(3)
        glued = glue_open_edges(g)
        reference = evaluate(glued)
        got = Graph(g).evaluate_symbolic().evaluate_numeric()
        assert got == pytest.approx(abs(reference), rel=1e-9)


# ---------------------------------------------------------------------------
# 3. The defect this file was written to expose
# ---------------------------------------------------------------------------

class TestThetaNormalisationDiscrepancy:
    """
    OPEN QUESTION, not yet adjudicated.

    src/ and the strand expansion agree on theta whenever every pair count
    m = j_a+j_b-j_c (and cyclic) is 0 or 1, and disagree otherwise, by exactly
    (m! n! p!)**2:

        theta(1,1,1):  src 24    strand 24     m,n,p = 1,1,1  -> factor 1
        theta(1,1,2):  src 30    strand 480    m,n,p = 0,2,2  -> factor 16
        theta(1.5,1.5,1): src 60 strand 240    m,n,p = 2,1,1  -> factor 4

    Every network verified so far (theta j=1, tetrahedron j=1, cube j=1) has
    all vertices at m=n=p=1, which is why the conventions coincide there and
    the cube value 6144 is agreed.  Which convention is intended for the
    general case has not been settled, so this test only PINS the relationship
    rather than asserting a winner.
    """

    @pytest.mark.parametrize("labels,factor", [
        ((1.0, 1.0, 1.0), 1),
        ((1.0, 1.0, 2.0), 16),
        ((1.5, 1.5, 1.0), 4),
        ((2.0, 2.0, 2.0), (2 * 2 * 2) ** 2),
    ])
    def test_relationship_between_the_two_conventions(self, labels, factor):
        from src.spin_evaluator import SpinNetworkEvaluator
        ev = SpinNetworkEvaluator(max_two_j=40, backend="serial", verbose=False)
        try:
            sign_exp, mag = ev.theta_symbol(*labels)
            src_theta = ((-1.0) ** int(round(sign_exp))) * mag
        finally:
            ev.cleanup()
        strand = evaluate(theta_net(labels))
        assert abs(strand) == pytest.approx(abs(src_theta) * factor, rel=1e-9), (
            f"theta{labels}: src={src_theta}, strand={strand}, "
            f"expected ratio {factor}"
        )


class TestKnownDefect:
    """
    The reduction is not independent of the path it takes.

    ``pick_smallest_interior_face_gt3`` chooses ``min(cycles, key=len)``.  When
    several faces share the shortest length -- which is the normal case -- the
    winner is whichever ``list_face_cycles`` emitted first, i.e. traversal
    order.  Different choices give different norms, so the result depends on
    node naming rather than on the graph.

    Ground truth for the 4-ring is 6144, confirmed three independent ways:
    the strand oracle below, a hand derivation via
    ``Theta(1,1,1)^4 * sum_F W6j(1,1,1,1,1,F)^4 * Delta_F = 331776/54``, and
    one of the code's own face choices.  The default choice returns 4608.
    """

    @pytest.mark.xfail(
        reason="reduction is path-dependent: the 4-ring gives 4608 on the "
               "default face choice and 6144 on another; 6144 is correct",
        strict=True,
    )
    def test_four_ring_matches_the_oracle(self):
        from src.gluer import glue_open_edges
        g = ring_with_legs(4)
        reference = abs(evaluate(glue_open_edges(g)))
        assert reference == pytest.approx(6144.0)
        got = Graph(g).evaluate_symbolic().evaluate_numeric()
        assert got == pytest.approx(reference, rel=1e-9)

    @pytest.mark.xfail(
        reason="same path-dependence, seen directly: forcing the F-move onto "
               "different faces of one fixed graph changes the norm",
        strict=True,
    )
    def test_norm_is_independent_of_which_face_the_f_move_uses(self):
        import src.graph_reducer as graph_reducer

        original = graph_reducer.pick_smallest_interior_face_gt3
        values = set()
        try:
            for k in range(4):
                def pick(cycles, k=k):
                    big = [c for c in cycles if len(c) > 3]
                    return big[k % len(big)] if big else None
                graph_reducer.pick_smallest_interior_face_gt3 = pick
                values.add(round(
                    Graph(ring_with_legs(4)).evaluate_symbolic().evaluate_numeric(), 6
                ))
        finally:
            graph_reducer.pick_smallest_interior_face_gt3 = original

        assert len(values) == 1, (
            f"the norm depends on which face the F-move is applied to: {sorted(values)}"
        )
