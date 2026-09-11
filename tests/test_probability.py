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
Tests for src/probability.py -- the transition-probability formula.

The quantity built here is

    P = norm(G_out) x (Delta/Theta factor from reconnections)
        ------------------------------------------------------
              norm(G_in)  x  norm(G_Delta)

with the "0 in, 0 out" convention: a norm or Theta that is exactly zero means
the state or reconnection is physically forbidden, so the factor contributes 0
rather than raising a division error.

WHY THIS MATTERS
----------------
This is the least-tested and most assembled part of the library: it stitches
three independently-reduced formulas plus a reconnection factor into one
string, then evaluates it. A mistake in the assembly (a factor on the wrong
side of the division, a lost reconnection) produces a number that looks
entirely reasonable.

Transitions are built headlessly here -- see tests/test_evolution.py for why.

The strongest test here is TestNormalisation: summed over every admissible
reconnection channel c, the probabilities must come to exactly 1. That single
sum rule exercises the norms, the Delta(c)/Theta(c,s,t) factor and the
direction of the ratio all at once.
"""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import Formula, Graph, UnitArg, SpinNetwork, calculate_probability
from src.evolution import LineageError, Transition
from src.probability import _delta_theta_factor_string, _label_literal
from src.spin_evaluator import FormulaEvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def closed_theta(labels=(1.0, 1.0, 2.0)):
    """Two vertices joined by three parallel edges. Norm = Theta(labels)^2."""
    g = nx.MultiGraph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(1.0, 0.0))
    for label in labels:
        g.add_edge(0, 1, label=label)
    return Graph(g)


def link(parent, child_graph, added=None, triplets=(), old_labels=()):
    """Attach a child through a Transition; the headless form of transition_to()."""
    t = Transition(
        parent=parent,
        added_graph=added if added is not None else Graph(nx.MultiGraph()),
        produced_open_ends=frozenset(),
        theta_triplets=tuple(triplets),
        old_open_end_labels=tuple(old_labels),
    )
    child = SpinNetwork(child_graph, parent_transition=t)
    t._link_child(child)
    parent._add_child_transition(t)
    return t, child


@pytest.fixture(scope="module")
def fev():
    """One shared evaluator: wigxjpf keeps process-global C tables."""
    evaluator = FormulaEvaluator(max_two_j=60, backend="serial", verbose=False)
    yield evaluator
    evaluator.cleanup()


# ===========================================================================
# _label_literal -- embedding an edge label into a formula string
# ===========================================================================

class TestLabelLiteral:

    def test_float_becomes_a_float_literal(self):
        assert _label_literal(1.5) == repr(1.5)

    def test_int_is_promoted_to_float(self):
        """Spins are half-integers; keeping ints would risk integer division."""
        assert _label_literal(2) == repr(2.0)

    def test_zero_is_rendered_not_dropped(self):
        """Spin 0 is a legitimate label and is falsy in Python."""
        assert _label_literal(0) == repr(0.0)
        assert _label_literal(0.0) == repr(0.0)

    def test_numeric_string_is_parsed_to_a_float_literal(self):
        assert _label_literal("1.5") == repr(1.5)

    def test_symbolic_name_passes_through(self):
        assert _label_literal("j_1") == "j_1"

    def test_primed_name_is_sanitised_to_a_python_identifier(self):
        """Gluing creates primed node names; formula strings must stay valid Python."""
        out = _label_literal("n''")
        assert "'" not in out
        compile(out, "<test>", "eval")   # raises if not a valid expression

    def test_output_is_always_valid_python(self):
        for label in (1.5, 2, 0, "1.5", "j_1", "n'", "n''"):
            compile(_label_literal(label), "<test>", "eval")


# ===========================================================================
# _delta_theta_factor_string -- the reconnection factor
# ===========================================================================

class TestDeltaThetaFactorString:

    def test_nothing_to_multiply_gives_the_identity(self):
        """Must be "1", not "" -- the result is substituted into a product."""
        assert _delta_theta_factor_string((), ()) == "1"

    def test_one_triplet_produces_one_deltatheta(self):
        out = _delta_theta_factor_string(((1.0, 1.0, 2.0),), ())
        assert out == "deltatheta(1.0, 1.0, 2.0)"

    def test_triplets_are_multiplied_together(self):
        out = _delta_theta_factor_string(((1.0, 1.0, 2.0), (0.5, 0.5, 1.0)), ())
        assert out == "deltatheta(1.0, 1.0, 2.0) * deltatheta(0.5, 0.5, 1.0)"

    def test_consumed_open_ends_contribute_delta_factors(self):
        out = _delta_theta_factor_string((), (1.0, 1.5))
        assert out == "delta(1.0) * delta(1.5)"

    def test_consumed_open_ends_and_reconnections_combine(self):
        """Both kinds of factor appear together when both are present."""
        out = _delta_theta_factor_string(((1.0, 1.0, 2.0),), (1.5,))
        assert "deltatheta(1.0, 1.0, 2.0)" in out
        assert "delta(1.5)" in out

    def test_symbolic_labels_are_kept_symbolic(self):
        out = _delta_theta_factor_string((("a", "b", "c"),), ("j_1",))
        assert out == "deltatheta(a, b, c) * delta(j_1)"

    def test_result_is_a_valid_python_expression(self):
        out = _delta_theta_factor_string(((1.0, 1.0, 2.0), ("a", "b", "c")), (1.5,))
        compile(out, "<test>", "eval")

    def test_evaluates_to_delta_over_theta(self, fev):
        """deltatheta(c,s,t) must be Delta(c) / Theta(c,s,t)."""
        expr = _delta_theta_factor_string(((1.0, 1.0, 2.0),), ())
        got = fev.evaluate(expr)
        expected = fev.evaluate("delta(1.0) / theta(1.0, 1.0, 2.0)")
        assert got == pytest.approx(expected, rel=1e-12)

    def test_inadmissible_reconnection_contributes_zero(self, fev):
        """
        (1, 1, 5) violates the triangle inequality, so Theta is zero. The
        convention is that the factor becomes 0, not a ZeroDivisionError.
        """
        expr = _delta_theta_factor_string(((1.0, 1.0, 5.0),), ())
        assert fev.evaluate(expr) == 0.0


# ===========================================================================
# calculate_probability -- assembly
# ===========================================================================

class TestCalculateProbability:

    def test_requires_an_ancestor_descendant_pair(self):
        n1 = SpinNetwork(closed_theta())
        stranger = SpinNetwork(closed_theta())
        with pytest.raises(LineageError):
            calculate_probability(n1, stranger)

    def test_returns_a_formula(self):
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        assert isinstance(calculate_probability(n1, n2), Formula)

    def test_builds_a_safe_div_expression(self):
        """
        The division must go through safe_div so a physically forbidden
        denominator gives 0 instead of raising.
        """
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        assert calculate_probability(n1, n2)._formula_string.startswith("safe_div(")

    def test_ratio_of_identical_states_is_one(self, fev):
        """
        With G_in == G_out, no added edges and no reconnections, P reduces to
        norm/norm = 1. This is the cleanest check that numerator and
        denominator are not swapped or mismatched.
        """
        n1 = SpinNetwork(closed_theta((1.0, 1.0, 2.0)))
        _, n2 = link(n1, closed_theta((1.0, 1.0, 2.0)))
        assert calculate_probability(n1, n2).evaluate_numeric() == pytest.approx(1.0)

    def test_numerator_is_the_child_and_denominator_the_parent(self, fev):
        """
        Direction matters. With different nets, P must be
        norm(out)/norm(in) -- so swapping the arguments must give the
        reciprocal, not the same number.
        """
        g_in = closed_theta((1.0, 1.0, 2.0))     # norm = Theta(1,1,2)^2 = 900
        g_out = closed_theta((1.0, 1.0, 1.0))    # norm = Theta(1,1,1)^2 = 576

        n1 = SpinNetwork(g_in)
        _, n2 = link(n1, g_out)
        forward = calculate_probability(n1, n2).evaluate_numeric()
        assert forward == pytest.approx(576.0 / 900.0, rel=1e-9)

    def test_reconnection_factor_enters_the_numerator(self, fev):
        """
        A reconnection multiplies the numerator by Delta(c)/Theta(c,s,t).
        Compare against the same transition without the triplet.
        """
        n1 = SpinNetwork(closed_theta())
        _, plain = link(n1, closed_theta())
        base = calculate_probability(n1, plain).evaluate_numeric()

        m1 = SpinNetwork(closed_theta())
        _, reconnected = link(m1, closed_theta(), triplets=((1.0, 1.0, 2.0),))
        with_factor = calculate_probability(m1, reconnected).evaluate_numeric()

        ratio = abs(fev.evaluate("delta(1.0) / theta(1.0, 1.0, 2.0)"))
        assert with_factor == pytest.approx(base * ratio, rel=1e-9)

    def test_added_edges_enter_the_denominator(self, fev):
        """
        G_Delta divides the result. Adding a closed theta subgraph must divide
        by its norm; with no added edges the factor is exactly 1.
        """
        added = nx.MultiGraph()
        added.add_node("p", pos=(0.0, 0.0))
        added.add_node("q", pos=(1.0, 0.0))
        for label in (1.0, 1.0, 1.0):
            added.add_edge("p", "q", label=label)

        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta(), added=Graph(added))
        with_delta = calculate_probability(n1, n2).evaluate_numeric()

        m1 = SpinNetwork(closed_theta())
        _, m2 = link(m1, closed_theta())
        without_delta = calculate_probability(m1, m2).evaluate_numeric()

        # norm of the added closed theta net = Theta(1,1,1)^2 = 576
        assert with_delta == pytest.approx(without_delta / 576.0, rel=1e-9)

    def test_no_added_edges_means_a_delta_factor_of_one(self):
        """An empty structural delta must contribute the literal "1"."""
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        assert calculate_probability(n1, n2)._formula_string.rstrip().endswith("(1))")

    def test_forbidden_reconnection_gives_probability_zero(self):
        """(1, 1, 5) is not a triangle: the transition cannot happen."""
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta(), triplets=((1.0, 1.0, 5.0),))
        assert calculate_probability(n1, n2).evaluate_numeric() == 0.0

    def test_result_is_never_negative(self):
        """A probability is handed back through the same abs() boundary as a norm."""
        n1 = SpinNetwork(closed_theta((1.0, 1.0, 1.0)))
        _, n2 = link(n1, closed_theta((1.0, 1.0, 2.0)))
        assert calculate_probability(n1, n2).evaluate_numeric() >= 0.0

    def test_multi_hop_path_composes_every_reconnection(self, fev):
        """
        Over n1 -> n2 -> n3, both reconnection factors must appear. Losing one
        silently rescales the probability.
        """
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta(), triplets=((1.0, 1.0, 2.0),))
        _, n3 = link(n2, closed_theta(), triplets=((0.5, 0.5, 1.0),))

        text = calculate_probability(n1, n3)._formula_string
        assert "deltatheta(1.0, 1.0, 2.0)" in text
        assert "deltatheta(0.5, 0.5, 1.0)" in text

    def test_multi_hop_uses_the_endpoints_not_the_intermediate_state(self, fev):
        """P(n1 -> n3) must divide by norm(n1), not norm(n2)."""
        n1 = SpinNetwork(closed_theta((1.0, 1.0, 2.0)))      # norm 900
        _, n2 = link(n1, closed_theta((1.0, 1.0, 1.0)))      # norm 576
        _, n3 = link(n2, closed_theta((1.0, 1.0, 2.0)))      # norm 900

        assert calculate_probability(n1, n3).evaluate_numeric() == \
               pytest.approx(900.0 / 900.0, rel=1e-9)

    def test_symbolic_labels_are_exposed_as_free_arguments(self):
        n1 = SpinNetwork(closed_theta(("a", "a", "b")))
        _, n2 = link(n1, closed_theta(("a", "a", "b")))
        formula = calculate_probability(n1, n2)
        assert {arg.label for arg in formula.get_args()} == {"a", "b"}

    def test_symbolic_probability_evaluates_with_assigned_spins(self):
        n1 = SpinNetwork(closed_theta(("a", "a", "b")))
        _, n2 = link(n1, closed_theta(("a", "a", "b")))
        formula = calculate_probability(n1, n2)
        value = formula.evaluate_numeric([UnitArg("a", 1.0), UnitArg("b", 2.0)])
        assert value == pytest.approx(1.0, rel=1e-9)

    def test_unassigned_variables_are_reported(self):
        n1 = SpinNetwork(closed_theta(("a", "a", "b")))
        _, n2 = link(n1, closed_theta(("a", "a", "b")))
        with pytest.raises(ValueError, match="unassigned"):
            calculate_probability(n1, n2).evaluate_numeric()

    def test_batch_evaluation_matches_one_by_one(self):
        n1 = SpinNetwork(closed_theta(("a", "a", "b")))
        _, n2 = link(n1, closed_theta(("a", "a", "b")))
        formula = calculate_probability(n1, n2)

        args_list = [[UnitArg("a", a), UnitArg("b", 2.0)] for a in (1.0, 1.5, 2.0)]
        batch = formula.evaluate_batch(args_list)
        one_by_one = [formula.evaluate_numeric(a) for a in args_list]
        assert batch == pytest.approx(one_by_one, rel=1e-12)

    def test_comparing_a_network_with_itself_is_refused_clearly(self):
        """
        lineage_to(self) is empty, so there is no transition to compose. This
        used to fail with a bare IndexError from transitions[0]; it now reports
        the actual problem.
        """
        n1 = SpinNetwork(closed_theta())
        with pytest.raises(LineageError, match="compared with itself"):
            calculate_probability(n1, n1)


# ===========================================================================
# The GUI path: metadata JSON -> Transition -> probability
# ===========================================================================

class TestGuiProducedTransition:
    """
    scripts/transition_to.py writes a JSON file with exactly two keys,
    'added_edges' and 'reconnections'. SpinNetwork.transition_from_metadata()
    is the parsing step that transition_to() performs once the GUI closes, so
    these tests cover the real GUI path without needing a display.

    The point of this class is to pin down which factors actually survive from
    a GUI transition into the final probability.
    """

    def gui_payload(self, reconnections=(), added_edges=()):
        """A metadata dict in the exact schema scripts/transition_to.py emits."""
        return {
            "added_edges": [
                {"nodes": list(e["nodes"]), "label": e["label"], "key": 0}
                for e in added_edges
            ],
            "reconnections": [
                {
                    "old_edges": [
                        {"nodes": [1, 2], "label": s},
                        {"nodes": [3, 4], "label": t},
                    ],
                    "new_edge": {"nodes": [5, 6], "label": c,
                                 "reconnection_node": 5},
                    "compute_all": False,
                }
                for (c, s, t) in reconnections
            ],
        }

    def test_reconnection_triplets_are_parsed_from_the_gui_json(self):
        n1 = SpinNetwork(closed_theta())
        n2 = n1.transition_from_metadata(
            closed_theta()._nx_graph,
            self.gui_payload(reconnections=[(2.0, 1.0, 1.0)]),
        )
        assert n2.parent_transition.theta_triplets == ((2.0, 1.0, 1.0),)

    def test_delta_over_theta_factor_reaches_the_probability_formula(self, fev):
        """
        THE KEY CHECK. A GUI-produced transition must contribute
        deltatheta(c, s, t) = Delta(c)/Theta(c, s, t) to the probability.
        """
        n1 = SpinNetwork(closed_theta())
        n2 = n1.transition_from_metadata(
            closed_theta()._nx_graph,
            self.gui_payload(reconnections=[(2.0, 1.0, 1.0)]),
        )
        formula = calculate_probability(n1, n2)
        assert "deltatheta(2.0, 1.0, 1.0)" in formula._formula_string

        # ... and it changes the number, not just the string.
        m1 = SpinNetwork(closed_theta())
        m2 = m1.transition_from_metadata(closed_theta()._nx_graph, self.gui_payload())
        ratio = abs(fev.evaluate("delta(2.0) / theta(2.0, 1.0, 1.0)"))
        assert formula.evaluate_numeric() == pytest.approx(
            calculate_probability(m1, m2).evaluate_numeric() * ratio, rel=1e-9
        )

    def test_every_reconnection_contributes_its_own_factor(self):
        n1 = SpinNetwork(closed_theta())
        n2 = n1.transition_from_metadata(
            closed_theta()._nx_graph,
            self.gui_payload(reconnections=[(2.0, 1.0, 1.0), (1.0, 0.5, 0.5)]),
        )
        text = calculate_probability(n1, n2)._formula_string
        assert "deltatheta(2.0, 1.0, 1.0)" in text
        assert "deltatheta(1.0, 0.5, 0.5)" in text

    def test_added_edges_become_the_g_delta_denominator(self):
        n1 = SpinNetwork(closed_theta())
        n2 = n1.transition_from_metadata(
            closed_theta()._nx_graph,
            self.gui_payload(added_edges=[
                {"nodes": ("p", "q"), "label": 1.0},
                {"nodes": ("p", "q"), "label": 1.0},
                {"nodes": ("p", "q"), "label": 1.0},
            ]),
        )
        assert n2.parent_transition.added_graph._nx_graph.number_of_edges() == 3
        # norm(G_delta) = Theta(1,1,1)^2 = 576 divides the result
        assert calculate_probability(n1, n2).evaluate_numeric() == \
               pytest.approx(1.0 / 576.0, rel=1e-9)

    def test_a_malformed_reconnection_is_refused_not_silently_dropped(self):
        """
        A reconnection whose record lacks a new_edge label used to be skipped
        by `if len(old_edges) == 2 and "label" in new_edge`, silently omitting
        its Delta/Theta factor. It must raise instead.
        """
        n1 = SpinNetwork(closed_theta())
        broken = {
            "added_edges": [],
            "reconnections": [{"old_edges": [{"label": 1.0}, {"label": 1.0}],
                               "new_edge": {}}],
        }
        with pytest.raises(ValueError, match="Malformed reconnection"):
            n1.transition_from_metadata(closed_theta()._nx_graph, broken)

    def test_a_closed_parent_consumes_no_open_ends(self):
        """
        The Delta(j) factor covers parent open ends closed off outside a
        reconnection. A parent with no open ends at all can have none, so the
        factor must be absent here -- see TestConsumedOpenEnds for the cases
        where it is present.
        """
        n1 = SpinNetwork(closed_theta())
        n2 = n1.transition_from_metadata(
            closed_theta()._nx_graph,
            self.gui_payload(reconnections=[(2.0, 1.0, 1.0)]),
        )
        assert n2.parent_transition.old_open_end_labels == ()

    def test_produced_open_ends_are_derived_from_the_two_graphs(self):
        parent = closed_theta()                      # closed: no open ends
        child_nx = nx.MultiGraph()
        child_nx.add_edge(0, 1, label=1.0)
        child_nx.add_edge(0, 1, label=1.0)
        child_nx.add_edge(0, "stub", label=1.0)
        for n in child_nx.nodes:
            child_nx.nodes[n]["pos"] = (0.0, 0.0)

        n1 = SpinNetwork(parent)
        n2 = n1.transition_from_metadata(child_nx, self.gui_payload())
        assert "stub" in n2.parent_transition.produced_open_ends


# ===========================================================================
# Delta(j): parent open ends consumed outside reconnections
# ===========================================================================

class TestConsumedOpenEnds:
    """
    A Delta(j) factor arises at each ATTACHMENT NODE: a node of the parent
    that an added edge connects to and that was not yet saturated
    (parent degree < 3).  Its parent-side edges are the open ends the
    transition consumes, and each contributes Delta(j).

    The same edges are also added to G_Delta (see
    SpinNetwork.transition_from_metadata step 2), so the Delta(j) numerator and
    the ||G_Delta|| denominator are always built from the same set.  Omitting
    step 2 leaves the attachment node non-trivalent and makes the probability
    too large -- for the recorded example, by a factor of 3.

    Open ends consumed by a RECONNECTION are excluded: they are already carried
    by that reconnection's Delta(c)/Theta(c,s,t) triplet, and a reconnection
    records no added_edges, so its nodes are never attachment nodes.
    """

    def parent_with_two_legs(self, s=1.0, t=1.0, m=1.0):
        """u--v (m), each carrying one open leg."""
        g = nx.MultiGraph()
        g.add_edge("u", "v", label=m)
        g.add_edge("u", "S", label=s)
        g.add_edge("v", "T", label=t)
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)
        return Graph(g)

    def test_open_end_consumed_by_added_edges_yields_a_delta_factor(self):
        """
        Leg S is built out into a trivalent vertex by two added edges, so the
        stub S is consumed. Its spin must appear as Delta(s).
        """
        parent = self.parent_with_two_legs(s=1.0, t=1.0)

        child = nx.MultiGraph()
        child.add_edge("u", "v", label=1.0)
        child.add_edge("v", "T", label=1.0)      # still open
        child.add_edge("u", "S", label=1.0)      # S now has more edges...
        child.add_edge("S", "A", label=1.0)      # ... so it is degree 3
        child.add_edge("S", "B", label=1.0)
        for n in child.nodes:
            child.nodes[n]["pos"] = (0.0, 0.0)

        meta = {
            "added_edges": [
                {"nodes": ["S", "A"], "label": 1.0, "key": 0},
                {"nodes": ["S", "B"], "label": 1.0, "key": 0},
            ],
            "reconnections": [],
        }
        n1 = SpinNetwork(parent)
        n2 = n1.transition_from_metadata(child, meta)
        assert n2.parent_transition.old_open_end_labels == (1.0,)

    def test_the_attachment_edge_is_included_in_g_delta(self):
        """
        G_Delta must contain the parent's edge at the attachment node, not just
        the drawn edges -- otherwise the attachment node stays non-trivalent
        and ||G_Delta|| is wrong.  Here that makes S a (1, 1, 1) vertex.
        """
        parent = self.parent_with_two_legs(s=1.0, t=1.0)
        meta = {
            "added_edges": [
                {"nodes": ["S", "A"], "label": 1.0, "key": 0},
                {"nodes": ["S", "B"], "label": 1.0, "key": 0},
            ],
            "reconnections": [],
        }
        n1 = SpinNetwork(parent)
        n2 = n1.transition_from_metadata(parent._nx_graph.copy(), meta)

        g_delta = n2.parent_transition.added_graph._nx_graph
        assert g_delta.degree("S") == 3, (
            f"attachment node S has degree {g_delta.degree('S')}, expected 3; "
            "the parent-side edge was not included in G_Delta"
        )
        # G_Delta is a Y with three open ends, so gluing it to its mirror
        # forms a theta net: ||G_Delta|| = |Theta(1,1,1)| = 24.  (Compare the
        # recorded example, where ||G_Delta|| = |Theta(0.5, 0.5, 1.0)| = 6.)
        assert n2.parent_transition.added_graph.evaluate_symbolic() \
                 .evaluate_numeric() == pytest.approx(24.0, rel=1e-9)

    def test_reconnected_open_ends_are_not_counted_twice(self):
        """
        Both legs vanish into a reconnection. They are already represented by
        deltatheta(c, s, t), so no Delta(j) may be emitted for them.
        """
        parent = self.parent_with_two_legs(s=1.0, t=1.0)

        child = nx.MultiGraph()
        child.add_edge("u", "v", label=1.0)
        child.add_edge("u", "w", label=1.0)
        child.add_edge("v", "w", label=1.0)
        child.add_edge("w", "C", label=2.0)      # the new channel
        for n in child.nodes:
            child.nodes[n]["pos"] = (0.0, 0.0)

        meta = {
            "added_edges": [],
            "reconnections": [{
                "old_edges": [{"nodes": ["S", "u"], "label": 1.0},
                              {"nodes": ["T", "v"], "label": 1.0}],
                "new_edge": {"nodes": ["w", "C"], "label": 2.0},
                "compute_all": False,
            }],
        }
        n1 = SpinNetwork(parent)
        n2 = n1.transition_from_metadata(child, meta)

        assert n2.parent_transition.theta_triplets == ((2.0, 1.0, 1.0),)
        assert n2.parent_transition.old_open_end_labels == (), (
            "a reconnected open end was double-counted as a Delta(j) factor"
        )

    def test_integer_node_names_from_the_gui_match_string_names_from_graphml(self):
        """
        scripts/transition_to.py converts numeric node names to int;
        _load_graphml keeps them as str. The comparison must survive that.
        """
        g = nx.MultiGraph()
        g.add_edge("0", "1", label=1.0)
        g.add_edge("0", "2", label=1.0)          # node "2" is the open end
        g.add_edge("1", "3", label=1.0)
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)

        child = nx.MultiGraph()
        child.add_edge("0", "1", label=1.0)
        child.add_edge("1", "3", label=1.0)
        child.add_edge("0", "4", label=1.0)
        child.add_edge("4", "5", label=1.0)
        child.add_edge("4", "6", label=1.0)
        for n in child.nodes:
            child.nodes[n]["pos"] = (0.0, 0.0)

        meta = {
            "added_edges": [],
            "reconnections": [{
                # int node names, as the GUI writes them
                "old_edges": [{"nodes": [2, 0], "label": 1.0},
                              {"nodes": [3, 1], "label": 1.0}],
                "new_edge": {"nodes": [4, 5], "label": 1.0},
                "compute_all": False,
            }],
        }
        n1 = SpinNetwork(Graph(g))
        n2 = n1.transition_from_metadata(child, meta)
        assert n2.parent_transition.old_open_end_labels == (), (
            "int/str node-name mismatch caused a reconnected end to be "
            "double-counted"
        )

    def test_delta_factor_reaches_the_probability_formula(self):
        """
        Uses a fully trivalent parent, since this test evaluates the norms.
        Leg S carries spin 2; building it out into a vertex (2, 1, 1) consumes
        it, so delta(2.0) must appear in the probability.
        """
        parent_nx = nx.MultiGraph()
        parent_nx.add_edge("u", "v", label=1.0)
        parent_nx.add_edge("u", "S", label=2.0)
        parent_nx.add_edge("u", "P", label=1.0)
        parent_nx.add_edge("v", "T", label=1.0)
        parent_nx.add_edge("v", "Q", label=1.0)
        for n in parent_nx.nodes:
            parent_nx.nodes[n]["pos"] = (0.0, 0.0)

        child = nx.MultiGraph()
        child.add_edge("u", "v", label=1.0)
        child.add_edge("u", "S", label=2.0)
        child.add_edge("u", "P", label=1.0)
        child.add_edge("v", "T", label=1.0)
        child.add_edge("v", "Q", label=1.0)
        child.add_edge("S", "A", label=1.0)     # S is now a (2, 1, 1) vertex
        child.add_edge("S", "B", label=1.0)
        for n in child.nodes:
            child.nodes[n]["pos"] = (0.0, 0.0)

        meta = {
            "added_edges": [
                {"nodes": ["S", "A"], "label": 1.0, "key": 0},
                {"nodes": ["S", "B"], "label": 1.0, "key": 0},
            ],
            "reconnections": [],
        }
        n1 = SpinNetwork(Graph(parent_nx))
        n2 = n1.transition_from_metadata(child, meta)
        assert n2.parent_transition.old_open_end_labels == (2.0,)
        assert "delta(2.0)" in calculate_probability(n1, n2)._formula_string

    def test_nothing_consumed_means_no_delta_factor(self):
        parent = self.parent_with_two_legs()
        n1 = SpinNetwork(parent)
        n2 = n1.transition_from_metadata(
            self.parent_with_two_legs()._nx_graph,
            {"added_edges": [], "reconnections": []},
        )
        assert n2.parent_transition.old_open_end_labels == ()


# ===========================================================================
# Normalisation: the physical validation of the whole construction
# ===========================================================================

class TestNormalisation:
    """
    THE PHYSICS CHECK.

    A reconnection merges two open legs s and t at a new trivalent vertex
    carrying (s, t, c). Summing the transition probability over every
    admissible channel c must give exactly 1: the network has to go somewhere.

    This is the strongest single validation in the suite. It exercises the
    norms, the Delta(c)/Theta(c,s,t) factor, and the direction of the ratio
    simultaneously -- getting any of them wrong breaks the sum rule.

    The test network keeps two spectator legs (p, q) after the reconnection.
    That matters: reducing a network to a single open end creates a bridge,
    and a subgraph hanging off one edge vanishes unless its spin is zero.
    """

    def pos_all(self, g):
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)
        return g

    def g_in(self, m, s, t, p, q):
        """u--v (m); u carries open legs s and p, v carries t and q."""
        g = nx.MultiGraph()
        g.add_edge("u", "v", label=m)
        g.add_edge("u", "S", label=s)
        g.add_edge("u", "P", label=p)
        g.add_edge("v", "T", label=t)
        g.add_edge("v", "Q", label=q)
        return self.pos_all(g)

    def g_out(self, m, s, t, p, q, c):
        """Legs s and t merged at a new vertex w = (s, t, c); c is now open."""
        g = nx.MultiGraph()
        g.add_edge("u", "v", label=m)
        g.add_edge("u", "w", label=s)
        g.add_edge("v", "w", label=t)
        g.add_edge("w", "C", label=c)
        g.add_edge("u", "P", label=p)
        g.add_edge("v", "Q", label=q)
        return self.pos_all(g)

    @staticmethod
    def channels(x, y):
        """Every c with |x-y| <= c <= x+y, stepping by 1."""
        out, c = [], abs(x - y)
        while c <= x + y + 1e-9:
            out.append(c)
            c += 1.0
        return out

    @staticmethod
    def vertex_ok(a, b, c):
        total = a + b + c
        return abs(a - b) <= c <= a + b and abs(total - round(total)) < 1e-9

    def probabilities(self, m, s, t, p, q):
        """P(c) for every admissible channel of the s+t reconnection."""
        result = {}
        for c in self.channels(s, t):
            meta = {
                "added_edges": [],
                "reconnections": [{
                    "old_edges": [{"nodes": ["S", "u"], "label": s},
                                  {"nodes": ["T", "v"], "label": t}],
                    "new_edge": {"nodes": ["w", "C"], "label": c},
                    "compute_all": False,
                }],
            }
            parent = SpinNetwork(Graph(self.g_in(m, s, t, p, q)))
            child = parent.transition_from_metadata(
                self.g_out(m, s, t, p, q, c), meta
            )
            result[c] = calculate_probability(parent, child).evaluate_numeric()
        return result

    # (m, s, t, p, q) -- all vertices admissible in the parent
    CASES = [
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 2.0, 2.0),
        (2.0, 1.0, 1.0, 1.0, 1.0),
        (1.0, 0.5, 0.5, 0.5, 0.5),
        (1.0, 1.5, 1.5, 0.5, 0.5),
        (2.0, 1.0, 2.0, 1.0, 2.0),
    ]

    @pytest.mark.parametrize("m,s,t,p,q", CASES)
    def test_probabilities_sum_to_one(self, m, s, t, p, q):
        assert self.vertex_ok(m, s, p) and self.vertex_ok(m, t, q), \
            "test case has an inadmissible parent vertex"
        probs = self.probabilities(m, s, t, p, q)
        total = sum(probs.values())
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"sum rule violated for m={m} s={s} t={t} p={p} q={q}: "
            f"sum={total!r} from {probs}"
        )

    @pytest.mark.parametrize("m,s,t,p,q", CASES)
    def test_every_channel_probability_is_in_the_unit_interval(self, m, s, t, p, q):
        for c, value in self.probabilities(m, s, t, p, q).items():
            assert 0.0 <= value <= 1.0 + 1e-9, (
                f"P(c={c}) = {value!r} is not a probability "
                f"(m={m} s={s} t={t} p={p} q={q})"
            )

    def test_at_least_one_case_has_several_open_channels(self):
        """Guards against the sum rule passing trivially on one-channel cases."""
        widest = max(len(self.channels(s, t)) for _, s, t, _, _ in self.CASES)
        assert widest >= 3


# ===========================================================================
# Added-edge transitions: the path the sum-rule test does NOT cover
# ===========================================================================

class TestAddedEdgeTransition:
    """
    TestNormalisation exercises pure reconnections, where G_Delta is empty.
    That left the added-edges path unchecked, which is exactly where the
    G_Delta construction bug lived: building G_Delta from the drawn edges
    alone leaves the attachment node non-trivalent, so ||G_Delta|| is too
    small and P is correspondingly too large (a factor of 8 for this fixture,
    3 for the example recorded in transition_to_graph_transition.json).

    These tests check the assembly against pieces computed independently.
    """

    def pos(self, g):
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)
        return g

    def parent(self):
        """u--v (1); u carries open legs P and S, v carries Q and T."""
        g = nx.MultiGraph()
        g.add_edge("u", "v", label=1.0)
        g.add_edge("u", "P", label=1.0)
        g.add_edge("u", "S", label=1.0)
        g.add_edge("v", "Q", label=1.0)
        g.add_edge("v", "T", label=1.0)
        return self.pos(g)

    def child(self):
        """Leg S built out into a trivalent vertex by two added edges."""
        g = self.parent()
        g.add_edge("S", "A", label=1.0)
        g.add_edge("S", "B", label=1.0)
        return self.pos(g)

    META = {
        "added_edges": [
            {"nodes": ["S", "A"], "label": 1.0, "key": 0},
            {"nodes": ["S", "B"], "label": 1.0, "key": 0},
        ],
        "reconnections": [],
    }

    def build(self):
        n1 = SpinNetwork(Graph(self.parent()))
        n2 = n1.transition_from_metadata(self.child(), self.META)
        return n1, n2

    def test_g_delta_includes_the_parent_edge_at_the_attachment_node(self):
        n1, n2 = self.build()
        g_delta = n2.parent_transition.added_graph._nx_graph
        # MultiGraph edge endpoints come back in insertion order, so
        # normalise each pair before comparing.
        edges = sorted(tuple(sorted((str(u), str(v)))) + (d["label"],)
                       for u, v, d in g_delta.edges(data=True))
        assert edges == [("A", "S", 1.0), ("B", "S", 1.0), ("S", "u", 1.0)], (
            f"G_Delta should be the Y  A--S--B plus S--u, got {edges}"
        )
        assert g_delta.degree("S") == 3

    def test_probability_matches_independently_computed_pieces(self, fev):
        """
        P = ||G_out|| x Delta(j) / (||G_in|| x ||G_Delta||), with every factor
        evaluated separately and multiplied by hand.
        """
        n1, n2 = self.build()
        t = n2.parent_transition

        norm_in = n1.evaluate_symbolic().evaluate_numeric()
        norm_out = n2.evaluate_symbolic().evaluate_numeric()
        norm_delta = t.added_graph.evaluate_symbolic().evaluate_numeric()
        delta_j = abs(fev.evaluate("delta(1.0)"))

        assert norm_delta == pytest.approx(24.0)   # |Theta(1,1,1)|
        assert delta_j == pytest.approx(3.0)
        assert t.old_open_end_labels == (1.0,)

        hand = norm_out * delta_j / (norm_in * norm_delta)
        assert calculate_probability(n1, n2).evaluate_numeric() == \
               pytest.approx(hand, rel=1e-12)

    def test_probability_is_one_for_this_transition(self):
        """
        1536 x 3 / (192 x 24) = 1 exactly.  Before the G_Delta fix this
        returned 8.0, because ||G_Delta|| was computed as 3 (the path A--S--B,
        whose middle node has degree 2) instead of 24.
        """
        n1, n2 = self.build()
        assert calculate_probability(n1, n2).evaluate_numeric() == \
               pytest.approx(1.0, rel=1e-9)

    def test_delta_j_and_g_delta_are_built_from_the_same_edges(self):
        """
        The Delta(j) numerator factors and the parent-side edges added to
        G_Delta must always be the same set -- they come from one loop over the
        attachment nodes. If they ever diverge, P is wrong by their ratio.
        """
        n1, n2 = self.build()
        t = n2.parent_transition
        g_delta = t.added_graph._nx_graph

        drawn = {("S", "A"), ("S", "B")}
        parent_side = [
            d["label"] for u, v, d in g_delta.edges(data=True)
            if tuple(sorted((str(u), str(v)))) not in
               {tuple(sorted(e)) for e in drawn}
        ]
        assert sorted(parent_side) == sorted(t.old_open_end_labels)
