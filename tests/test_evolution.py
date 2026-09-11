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
Tests for src/evolution.py -- the genealogy layer (Transition, LineageError)
and the lineage methods SpinNetwork exposes in src/api.py.

WHY THIS MATTERS
----------------
`calculate_probability(n_in, n_out)` starts by asking for the lineage path
between two networks and composing the transitions along it. If composition
loses a reconnection triplet, or if the lineage walk accepts a pair that is
not actually ancestor-descendant, the probability is computed from the wrong
structural data -- and it still returns a plausible-looking number.

Everything here is built headlessly. The only public way to create a
transition is `SpinNetwork.transition_to()`, which blocks on a Tkinter window,
so these tests assemble the same object graph directly via the constructors.
That is also the documented workaround for scripting transitions without a GUI.
"""

import os
import sys

import networkx as nx
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import Graph, SpinNetwork
from src.evolution import LineageError, Transition

# ---------------------------------------------------------------------------
# Helpers: build a genealogy without touching the GUI
# ---------------------------------------------------------------------------


def closed_theta(labels=(1.0, 1.0, 2.0)):
    """Two vertices joined by three parallel edges: the simplest closed net."""
    g = nx.MultiGraph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(1.0, 0.0))
    for label in labels:
        g.add_edge(0, 1, label=label)
    return Graph(g)


def open_net(stub_labels=(1.0, 1.0)):
    """A small net with degree-1 stub nodes, i.e. genuine open ends."""
    g = nx.MultiGraph()
    g.add_edge(0, 1, label=1.0)
    g.add_edge(0, 1, label=1.0)
    for i, label in enumerate(stub_labels):
        g.add_edge(i, f"s{i}", label=label)
    for n in g.nodes:
        g.nodes[n]["pos"] = (0.0, 0.0)
    return Graph(g)


def empty_graph():
    """An empty structural delta: a transition that only reconnected edges."""
    return Graph(nx.MultiGraph())


def one_edge_graph(label=1.0):
    g = nx.MultiGraph()
    g.add_edge("x", "y", label=label)
    for n in g.nodes:
        g.nodes[n]["pos"] = (0.0, 0.0)
    return Graph(g)


def link(parent, child_graph, added=None, open_ends=(), triplets=(), old_labels=()):
    """
    Attach a child to `parent` through a new Transition and return
    (transition, child) -- the headless equivalent of transition_to().
    """
    t = Transition(
        parent=parent,
        added_graph=added if added is not None else empty_graph(),
        produced_open_ends=frozenset(open_ends),
        theta_triplets=tuple(triplets),
        old_open_end_labels=tuple(old_labels),
    )
    child = SpinNetwork(child_graph, parent_transition=t)
    t._link_child(child)
    parent._add_child_transition(t)
    return t, child


# ===========================================================================
# LineageError
# ===========================================================================


class TestLineageError:

    def test_is_a_valueerror(self):
        """Callers catching ValueError must also catch this."""
        assert issubclass(LineageError, ValueError)


# ===========================================================================
# Transition construction and immutability
# ===========================================================================


class TestTransitionConstruction:

    def test_exposes_the_data_it_was_given(self):
        parent = SpinNetwork(closed_theta())
        added = one_edge_graph()
        t = Transition(
            parent=parent,
            added_graph=added,
            produced_open_ends={"s0", "s1"},
            theta_triplets=(("c", "s", "t"),),
            old_open_end_labels=(1.0, 1.5),
        )
        assert t.parent is parent
        assert t.added_graph is added
        assert t.produced_open_ends == frozenset({"s0", "s1"})
        assert t.theta_triplets == (("c", "s", "t"),)
        assert t.old_open_end_labels == (1.0, 1.5)

    def test_child_is_none_until_linked(self):
        t = Transition(SpinNetwork(closed_theta()), empty_graph(), frozenset())
        assert t.child is None

    def test_open_ends_are_normalised_to_a_frozenset(self):
        """A list must not stay mutable behind the property."""
        t = Transition(SpinNetwork(closed_theta()), empty_graph(), ["a", "b", "a"])
        assert t.produced_open_ends == frozenset({"a", "b"})
        assert isinstance(t.produced_open_ends, frozenset)

    def test_triplets_are_normalised_to_a_tuple(self):
        t = Transition(
            SpinNetwork(closed_theta()),
            empty_graph(),
            frozenset(),
            theta_triplets=[("c", "s", "t")],
        )
        assert isinstance(t.theta_triplets, tuple)

    def test_defaults_are_empty(self):
        t = Transition(SpinNetwork(closed_theta()), empty_graph(), frozenset())
        assert t.theta_triplets == ()
        assert t.old_open_end_labels == ()

    def test_link_child_sets_the_child(self):
        parent = SpinNetwork(closed_theta())
        t = Transition(parent, empty_graph(), frozenset())
        child = SpinNetwork(closed_theta(), parent_transition=t)
        t._link_child(child)
        assert t.child is child

    def test_link_child_twice_is_refused(self):
        """Transitions are immutable once linked; re-linking would corrupt the tree."""
        parent = SpinNetwork(closed_theta())
        t = Transition(parent, empty_graph(), frozenset())
        t._link_child(SpinNetwork(closed_theta(), parent_transition=t))
        with pytest.raises(LineageError, match="already has a child"):
            t._link_child(SpinNetwork(closed_theta()))

    def test_repr_is_readable_and_does_not_crash_when_unlinked(self):
        t = Transition(SpinNetwork(closed_theta()), empty_graph(), frozenset({"s0"}))
        text = repr(t)
        assert "Transition(" in text and "child=?" in text and "+1 open ends" in text


# ===========================================================================
# Transition.compose -- collapsing a multi-hop path
# ===========================================================================


class TestTransitionCompose:

    def test_spans_from_the_first_parent_to_the_last_child(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta())
        t2, n3 = link(n2, closed_theta())

        composed = t1.compose(t2)
        assert composed.parent is n1
        assert composed.child is n3

    def test_merges_both_structural_deltas(self):
        n1 = SpinNetwork(closed_theta())
        g_a = nx.MultiGraph()
        g_a.add_edge("p", "q", label=1.0)
        g_b = nx.MultiGraph()
        g_b.add_edge("r", "s", label=2.0)

        t1, n2 = link(n1, closed_theta(), added=Graph(g_a))
        t2, n3 = link(n2, closed_theta(), added=Graph(g_b))

        merged = t1.compose(t2).added_graph._nx_graph
        assert merged.number_of_edges() == 2
        assert set(merged.nodes()) == {"p", "q", "r", "s"}

    def test_concatenates_reconnection_triplets_in_order(self):
        """
        Every reconnection contributes a Delta/Theta factor to the probability,
        so losing or reordering one changes the answer.
        """
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta(), triplets=(("c1", "s1", "t1"),))
        t2, n3 = link(n2, closed_theta(), triplets=(("c2", "s2", "t2"),))

        assert t1.compose(t2).theta_triplets == (
            ("c1", "s1", "t1"),
            ("c2", "s2", "t2"),
        )

    def test_concatenates_consumed_open_end_labels(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta(), old_labels=(1.0,))
        t2, n3 = link(n2, closed_theta(), old_labels=(1.5, 2.0))
        assert t1.compose(t2).old_open_end_labels == (1.0, 1.5, 2.0)

    def test_net_open_ends_are_computed_from_the_endpoint_graphs(self):
        """
        When both endpoints are available, compose() compares their degree-1
        nodes directly rather than unioning the per-step sets -- so an open end
        created in step 1 and consumed in step 2 must not appear.
        """
        n1 = SpinNetwork(closed_theta())  # no open ends
        t1, n2 = link(n1, open_net(), open_ends={"s0", "s1"})
        t2, n3 = link(n2, closed_theta(), open_ends=set())  # back to closed

        composed = t1.compose(t2)
        assert (
            composed.produced_open_ends == frozenset()
        ), "an open end created then consumed must not survive composition"

    def test_net_open_ends_reports_ends_present_in_the_final_state(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta())
        t2, n3 = link(n2, open_net())
        assert t1.compose(t2).produced_open_ends == frozenset({"s0", "s1"})

    def test_falls_back_to_the_union_when_the_child_is_unlinked(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta(), open_ends={"a"})
        t2 = Transition(n2, empty_graph(), frozenset({"b"}))  # never linked
        assert t1.compose(t2).produced_open_ends == frozenset({"a", "b"})

    def test_composing_three_hops_left_to_right_accumulates_everything(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta(), triplets=(("c1", "s1", "t1"),))
        t2, n3 = link(n2, closed_theta(), triplets=(("c2", "s2", "t2"),))
        t3, n4 = link(n3, closed_theta(), triplets=(("c3", "s3", "t3"),))

        composed = t1.compose(t2).compose(t3)
        assert composed.parent is n1
        assert composed.child is n4
        assert len(composed.theta_triplets) == 3

    def test_does_not_mutate_either_input(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta(), triplets=(("c1", "s1", "t1"),))
        t2, n3 = link(n2, closed_theta(), triplets=(("c2", "s2", "t2"),))

        t1.compose(t2)
        assert t1.theta_triplets == (("c1", "s1", "t1"),)
        assert t2.theta_triplets == (("c2", "s2", "t2"),)
        assert t1.child is n2 and t2.child is n3


# ===========================================================================
# SpinNetwork genealogy: lineage_to, children, depth
# ===========================================================================


class TestSpinNetworkLineage:

    def test_root_has_no_parent_transition(self):
        assert SpinNetwork(closed_theta()).parent_transition is None

    def test_lineage_to_self_is_empty(self):
        n1 = SpinNetwork(closed_theta())
        assert n1.lineage_to(n1) == []

    def test_lineage_to_direct_child_is_one_hop(self):
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta())
        assert n1.lineage_to(n2) == [t1]

    def test_lineage_is_ordered_parent_first(self):
        """The path must read n1 -> n2 -> n3, not reversed."""
        n1 = SpinNetwork(closed_theta())
        t1, n2 = link(n1, closed_theta())
        t2, n3 = link(n2, closed_theta())
        assert n1.lineage_to(n3) == [t1, t2]

    def test_lineage_to_an_unrelated_network_is_refused(self):
        n1 = SpinNetwork(closed_theta())
        stranger = SpinNetwork(closed_theta())
        with pytest.raises(LineageError, match="not a descendant"):
            n1.lineage_to(stranger)

    def test_lineage_in_the_wrong_direction_is_refused(self):
        """A child is not an ancestor of its parent."""
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        with pytest.raises(LineageError):
            n2.lineage_to(n1)

    def test_lineage_across_sibling_branches_is_refused(self):
        n1 = SpinNetwork(closed_theta())
        _, left = link(n1, closed_theta())
        _, right = link(n1, closed_theta())
        with pytest.raises(LineageError):
            left.lineage_to(right)

    def test_children_lists_every_outgoing_transition(self):
        n1 = SpinNetwork(closed_theta())
        t_left, _ = link(n1, closed_theta())
        t_right, _ = link(n1, closed_theta())
        assert set(n1.children) == {t_left, t_right}

    def test_children_is_an_immutable_view(self):
        """Mutating the returned tuple must not corrupt the tree."""
        n1 = SpinNetwork(closed_theta())
        link(n1, closed_theta())
        assert isinstance(n1.children, tuple)
        before = len(n1.children)
        with pytest.raises(AttributeError):
            n1.children.append("nonsense")  # tuples have no append
        assert len(n1.children) == before

    def test_genealogy_depth_counts_transitions_from_the_root(self):
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        _, n3 = link(n2, closed_theta())
        assert n1._genealogy_depth() == 0
        assert n2._genealogy_depth() == 1
        assert n3._genealogy_depth() == 2

    def test_branches_are_independent(self):
        """Two children of the same parent are both at depth 1."""
        n1 = SpinNetwork(closed_theta())
        _, left = link(n1, closed_theta())
        _, right = link(n1, closed_theta())
        assert left._genealogy_depth() == right._genealogy_depth() == 1
        assert left is not right

    def test_each_network_gets_a_distinct_id(self):
        ids = {SpinNetwork(closed_theta())._id for _ in range(20)}
        assert len(ids) == 20

    def test_repr_reports_depth(self):
        n1 = SpinNetwork(closed_theta())
        _, n2 = link(n1, closed_theta())
        assert "depth=0" in repr(n1)
        assert "depth=1" in repr(n2)
