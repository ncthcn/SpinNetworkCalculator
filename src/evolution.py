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
Evolutionary genealogy layer for SpinNetworkCalculator.

Provides the Transition class (the "edge" in the genealogy tree) and the
LineageError exception.  The SpinNetwork class (the "node") lives in api.py
and imports from here; circular imports are avoided by using TYPE_CHECKING
guards so that evolution.py never imports from api.py at runtime.

Design
------
The genealogy forms a directed tree:
  - Each SpinNetwork has exactly one parent_transition (or None for the root).
  - Each Transition links one parent SpinNetwork to one child SpinNetwork.
  - The child link is filled exactly once via _link_child() after the child
    SpinNetwork is instantiated (construction-time circular dependency).
  - Multi-generational paths are handled by compose().
"""

from __future__ import annotations

from typing import TYPE_CHECKING, FrozenSet, List, Optional, Tuple

# TYPE_CHECKING is False at runtime, True for type-checkers only.
# This avoids the circular import api.py → evolution.py → api.py.
if TYPE_CHECKING:
    from src.api import Graph, SpinNetwork


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LineageError(ValueError):
    """
    Raised when two SpinNetworks do not share a valid ancestor–descendant
    lineage, making probability computation undefined.

    Example
    -------
        calculate_probability(n_a, n_b)  # raises if n_b is not a descendant of n_a
    """


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


class Transition:
    """
    A first-class entity representing the structural delta between a parent
    SpinNetwork and its child (the "edge" in the genealogy tree).

    Attributes
    ----------
    parent : SpinNetwork
        The SpinNetwork state before this transition.
    child : SpinNetwork or None
        The SpinNetwork state after this transition.  Set via _link_child()
        after the child is constructed.
    added_graph : Graph
        The new subgraph attached to the parent during this transition
        (edges explicitly added by the user).  Empty graph if only
        reconnections were performed with no new edges.
    produced_open_ends : frozenset of str
        Node labels of degree-1 stub nodes in the child that did not exist
        as open ends in the parent.  Used by compose() and calculate_probability()
        to track which open ends survive across multiple generations.

    Note on probability
    --------------------
    There is no pre-computed/cached probability field: the Transition Tool
    GUI (scripts/transition_to.py) only records structural metadata (added
    edges, reconnections), not norms.  calculate_probability() always builds
    the symbolic probability Formula from theta_triplets / old_open_end_labels
    and the graphs themselves — see src/probability.py.
    """

    def __init__(
        self,
        parent: "SpinNetwork",
        added_graph: "Graph",
        produced_open_ends: FrozenSet[str],
        theta_triplets: Tuple[Tuple, ...] = (),
        old_open_end_labels: Tuple[float, ...] = (),
    ) -> None:
        """
        Parameters
        ----------
        parent : SpinNetwork
            The parent state.
        added_graph : Graph
            The structural delta (new edges only, may be empty).
        produced_open_ends : frozenset of str
            Node labels of new open-end stubs produced by this transition.
        theta_triplets : tuple of (c, s, t) tuples
            Reconnection triplets for the Δ/Θ factor in probability computation.
            Each entry is (new_label, old_label_1, old_label_2).
        old_open_end_labels : tuple of float
            Labels of the parent's open ends that were consumed (closed) by this
            transition, needed for the Δ_old factor.
        """
        self._parent: "SpinNetwork" = parent
        self._child: Optional["SpinNetwork"] = None
        self._added_graph: "Graph" = added_graph
        self._produced_open_ends: FrozenSet[str] = frozenset(produced_open_ends)
        self._theta_triplets: Tuple[Tuple, ...] = tuple(theta_triplets)
        self._old_open_end_labels: Tuple[float, ...] = tuple(old_open_end_labels)

    # ------------------------------------------------------------------
    # Properties (read-only public surface)
    # ------------------------------------------------------------------

    @property
    def parent(self) -> "SpinNetwork":
        """The SpinNetwork state before this transition."""
        return self._parent

    @property
    def child(self) -> Optional["SpinNetwork"]:
        """The SpinNetwork state after this transition, or None if not yet linked."""
        return self._child

    @property
    def added_graph(self) -> "Graph":
        """The structural delta: edges added during this transition."""
        return self._added_graph

    @property
    def produced_open_ends(self) -> FrozenSet[str]:
        """Node labels of new open-end stubs produced by this transition."""
        return self._produced_open_ends

    @property
    def theta_triplets(self) -> Tuple[Tuple, ...]:
        """Reconnection triplets (c, s, t) for probability recomputation."""
        return self._theta_triplets

    @property
    def old_open_end_labels(self) -> Tuple[float, ...]:
        """Labels of parent open ends consumed (closed) by this transition."""
        return self._old_open_end_labels

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _link_child(self, child: "SpinNetwork") -> None:
        """
        Set the child SpinNetwork exactly once.

        Called by SpinNetwork.transition_to() after the child is instantiated.
        Raises LineageError if called a second time (immutability guard).

        Parameters
        ----------
        child : SpinNetwork
            The newly created child state.
        """
        if self._child is not None:
            raise LineageError(
                "This Transition already has a child.  "
                "Transitions are immutable after the child is linked."
            )
        self._child = child

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose(self, other: "Transition") -> "Transition":
        """
        Compose two consecutive transitions into a single multi-generational one.

        If self is the transition A→B and other is B→C, the result represents
        A→C directly: the combined added_graph merges both deltas, and
        produced_open_ends is the net set of new open ends across both steps.

        Parameters
        ----------
        other : Transition
            Must be the direct successor of self (i.e. other.parent is self.child).

        Returns
        -------
        Transition
            A synthetic Transition spanning from self.parent to other.child.
            Its probability must be computed via calculate_probability().
        """
        import networkx as nx
        from src.api import (
            Graph,
        )  # deferred import avoids circular dependency at module load

        # Merge the two structural deltas
        merged_nx = nx.compose(
            self._added_graph._nx_graph,
            other._added_graph._nx_graph,
        )
        merged_graph = Graph(merged_nx)

        # Net produced open ends: compare parent (G_A) and child (G_C) directly
        # when both are available, for the most accurate result.
        if other._child is not None:
            parent_open = frozenset(
                str(n) for n, d in self._parent._graph._nx_graph.degree() if d == 1
            )
            child_open = frozenset(
                str(n) for n, d in other._child._graph._nx_graph.degree() if d == 1
            )
            net_open_ends: FrozenSet[str] = child_open - parent_open
        else:
            # Fallback when child is not yet linked
            net_open_ends = self._produced_open_ends | other._produced_open_ends

        # Combined reconnection data for probability recomputation
        combined_triplets = self._theta_triplets + other._theta_triplets
        combined_old_labels = self._old_open_end_labels + other._old_open_end_labels

        t = Transition(
            parent=self._parent,
            added_graph=merged_graph,
            produced_open_ends=net_open_ends,
            theta_triplets=combined_triplets,
            old_open_end_labels=combined_old_labels,
        )
        if other._child is not None:
            t._link_child(other._child)
        return t

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parent_id = (
            self._parent._id
            if hasattr(self._parent, "_id")
            else f"{id(self._parent):#x}"
        )
        child_id = (
            self._child._id
            if self._child is not None and hasattr(self._child, "_id")
            else "?"
        )
        return (
            f"Transition("
            f"parent={parent_id}, "
            f"child={child_id}, "
            f"+{len(self._produced_open_ends)} open ends"
            f")"
        )
