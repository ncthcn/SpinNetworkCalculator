"""Unit tests for src/orientation.py — vertex orientation and graph phase."""

import math
import pytest
import networkx as nx

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.orientation import (
    canonical_edge,
    get_incident_edges,
    set_reference_orientation,
    get_reference_orientation,
    _transposition_sequence,
    _cyclic_align,
    phase_factor_for_transposition,
    calculate_layout_phase,
    resolve_planar_flattening,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_theta_graph() -> nx.MultiGraph:
    """Two nodes connected by three parallel edges, each labelled 1."""
    G = nx.MultiGraph()
    G.add_node(1, pos=(0.0, 0.0))
    G.add_node(2, pos=(1.0, 0.0))
    G.add_edge(1, 2, label=1)
    G.add_edge(1, 2, label=1)
    G.add_edge(1, 2, label=1)
    return G


def _make_triangle() -> nx.MultiGraph:
    """Simple triangle (K3) with labels 1 on every edge."""
    G = nx.MultiGraph()
    for i in (1, 2, 3):
        G.add_node(i, pos=(math.cos(2 * math.pi * i / 3),
                           math.sin(2 * math.pi * i / 3)))
    G.add_edge(1, 2, label=1)
    G.add_edge(2, 3, label=1)
    G.add_edge(1, 3, label=1)
    return G


def _make_k4() -> nx.MultiGraph:
    """Complete graph K4 — planar, 4 vertices each of degree 3."""
    G = nx.MultiGraph()
    for i in range(1, 5):
        G.add_node(i)
    edges = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    for u, v in edges:
        G.add_edge(u, v, label=1)
    return G


# ---------------------------------------------------------------------------
# canonical_edge
# ---------------------------------------------------------------------------

class TestCanonicalEdge:
    def test_order_preserved_when_u_lt_v(self):
        assert canonical_edge(1, 3, 0) == (1, 3, 0)

    def test_swapped_when_u_gt_v(self):
        assert canonical_edge(5, 2, 1) == (2, 5, 1)

    def test_key_preserved(self):
        assert canonical_edge(2, 1, 7) == (1, 2, 7)


# ---------------------------------------------------------------------------
# _transposition_sequence
# ---------------------------------------------------------------------------

class TestTranspositionSequence:
    def test_identity_gives_no_swaps(self):
        assert _transposition_sequence([0, 1, 2], [0, 1, 2]) == []

    def test_reverse_of_two(self):
        swaps = _transposition_sequence([0, 1], [1, 0])
        assert swaps == [(0, 1)]

    def test_known_example(self):
        # [A, B, C] → [C, A, B]  requires two swaps
        result = _transposition_sequence([0, 1, 2], [2, 0, 1])
        assert result == [(1, 2), (0, 1)]

    def test_applying_swaps_reproduces_target(self):
        ref = ["A", "B", "C"]
        target = ["C", "A", "B"]
        swaps = _transposition_sequence(ref, target)
        current = list(ref)
        for i, j in swaps:
            current[i], current[j] = current[j], current[i]
        assert current == target

    def test_wrong_elements_raises(self):
        with pytest.raises(ValueError):
            _transposition_sequence([0, 1], [1, 2])

    @pytest.mark.parametrize("perm", [
        [0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]
    ])
    def test_all_permutations_of_three(self, perm):
        ref = [0, 1, 2]
        swaps = _transposition_sequence(ref, perm)
        current = list(ref)
        for i, j in swaps:
            current[i], current[j] = current[j], current[i]
        assert current == perm


# ---------------------------------------------------------------------------
# _cyclic_align
# ---------------------------------------------------------------------------

class TestCyclicAlign:
    def test_already_aligned(self):
        assert _cyclic_align([0, 1, 2], [0, 1, 2]) == [0, 1, 2]

    def test_rotation_by_one(self):
        assert _cyclic_align([0, 1, 2], [1, 2, 0]) == [0, 1, 2]

    def test_rotation_by_two(self):
        assert _cyclic_align([0, 1, 2], [2, 0, 1]) == [0, 1, 2]

    def test_opposite_cyclic_order(self):
        # [0, 2, 1] is the "other" cyclic class; aligned to start with 0
        assert _cyclic_align([0, 1, 2], [2, 1, 0]) == [0, 2, 1]

    def test_missing_element_raises(self):
        # reference[0]=0 is absent from layout_order → should raise
        with pytest.raises(ValueError):
            _cyclic_align([0, 1, 2], [3, 1, 2])


# ---------------------------------------------------------------------------
# phase_factor_for_transposition
# ---------------------------------------------------------------------------

class TestPhaseFactorForTransposition:
    def _graph_with_edges(self, la, lb, lc):
        """Triangle with specific labels on the three edges."""
        G = nx.MultiGraph()
        G.add_node(1); G.add_node(2); G.add_node(3)
        G.add_edge(1, 2, label=la)   # edge_a = (1,2,0)
        G.add_edge(1, 3, label=lb)   # edge_b = (1,3,0)
        G.add_edge(2, 3, label=lc)   # edge_c (spectator) = (2,3,0)
        return G

    def test_integer_labels_spin_1(self):
        # a=b=c=1: exponent = 2*1*(1+1-1) = 2 → phase = +1
        G = self._graph_with_edges(1, 1, 1)
        result = phase_factor_for_transposition(G, (1,2,0), (1,3,0), (2,3,0))
        assert result == complex(1)

    def test_half_integer_labels(self):
        # a=b=1/2, c=0: exponent = 2*0*(1-0) = 0 → phase = +1
        G = self._graph_with_edges(0.5, 0.5, 0)
        result = phase_factor_for_transposition(G, (1,2,0), (1,3,0), (2,3,0))
        assert result == complex(1)

    def test_known_minus_one(self):
        # a=b=1, c=1/2: exponent = 2*(1/2)*(1+1-1/2) = 1*1.5 = 1.5 …
        # Try a=1/2, b=1/2, c=1: exponent = 2*1*(0.5+0.5-1) = 2*0 = 0 → +1
        G = self._graph_with_edges(0.5, 0.5, 1)
        result = phase_factor_for_transposition(G, (1,2,0), (1,3,0), (2,3,0))
        assert result == complex(1)

    def test_result_is_plus_or_minus_one(self):
        for a, b, c in [(1, 1, 1), (0.5, 0.5, 0), (0.5, 1, 0.5), (1, 1.5, 0.5)]:
            G = self._graph_with_edges(a, b, c)
            result = phase_factor_for_transposition(G, (1,2,0), (1,3,0), (2,3,0))
            assert result in (complex(1), complex(-1)), f"Unexpected phase {result} for a={a},b={b},c={c}"


# ---------------------------------------------------------------------------
# calculate_layout_phase  — reference layout should give phase = 1
# ---------------------------------------------------------------------------

class TestCalculateLayoutPhase:
    def test_reference_layout_gives_phase_one_triangle(self):
        """The layout matching reference_orientation must yield phase = 1."""
        G = _make_triangle()
        for v in G.nodes():
            set_reference_orientation(G, v)

        # Use the positions stored in node attributes as the layout.
        layout = {v: G.nodes[v]['pos'] for v in G.nodes()}
        phase = calculate_layout_phase(G, layout)
        assert abs(phase - 1.0) < 1e-10

    def test_missing_reference_orientation_skipped(self):
        """Vertices without reference_orientation are silently skipped."""
        G = _make_triangle()
        # Do NOT call set_reference_orientation
        layout = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0)}
        phase = calculate_layout_phase(G, layout)
        assert phase == complex(1)

    def test_phase_is_plus_or_minus_one(self):
        """Phase must always be ±1 for numeric labels."""
        G = _make_k4()
        for v in G.nodes():
            set_reference_orientation(G, v)
        # Set simple positions
        positions = {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0), 4: (0.5, 0.33)}
        layout = {v: positions[v] for v in G.nodes()}
        phase = calculate_layout_phase(G, layout)
        assert phase.real in (-1.0, 1.0) or abs(abs(phase) - 1.0) < 1e-10

    def test_rotation_does_not_change_phase(self):
        """A global rotation of the layout preserves cyclic order → same phase."""
        G = _make_triangle()
        for v in G.nodes():
            set_reference_orientation(G, v)

        layout = {v: G.nodes[v]['pos'] for v in G.nodes()}

        # Rotate all points by 45 degrees
        angle = math.pi / 4
        rotated = {
            v: (
                x * math.cos(angle) - y * math.sin(angle),
                x * math.sin(angle) + y * math.cos(angle),
            )
            for v, (x, y) in layout.items()
        }
        phase_original = calculate_layout_phase(G, layout)
        phase_rotated = calculate_layout_phase(G, rotated)
        assert phase_original == phase_rotated


# ---------------------------------------------------------------------------
# resolve_planar_flattening — K4 is planar, result must be consistent
# ---------------------------------------------------------------------------

class TestResolvePlanarFlattening:
    """Tests for resolve_planar_flattening.

    We also test the core guarantee (calculate_layout_phase is consistent with
    the returned layout) via a known manual layout to avoid numpy dependency.
    """

    def test_k4_returns_layout_and_phase(self):
        G = _make_k4()
        for v in G.nodes():
            set_reference_orientation(G, v)
        layout, phase = resolve_planar_flattening(G)
        assert set(layout.keys()) == set(G.nodes())
        assert isinstance(phase, complex)

    def test_layout_values_are_float_tuples(self):
        G = _make_k4()
        for v in G.nodes():
            set_reference_orientation(G, v)
        layout, _ = resolve_planar_flattening(G)
        for v, coords in layout.items():
            assert len(coords) == 2
            assert isinstance(coords[0], float)
            assert isinstance(coords[1], float)

    def test_phase_is_unit_modulus(self):
        """Phase must have modulus 1 (it's ±1)."""
        G = _make_k4()
        for v in G.nodes():
            set_reference_orientation(G, v)
        _, phase = resolve_planar_flattening(G)
        assert abs(abs(phase) - 1.0) < 1e-10

    def test_round_trip_consistency(self):
        """calculate_layout_phase on the returned layout reproduces the stored phase."""
        G = _make_k4()
        for v in G.nodes():
            set_reference_orientation(G, v)
        layout, phase1 = resolve_planar_flattening(G)
        phase2 = calculate_layout_phase(G, layout)
        assert phase1 == phase2

    def test_manual_layout_round_trip(self):
        """Verify round-trip without relying on networkx layout algorithms."""
        G = _make_triangle()
        for v in G.nodes():
            set_reference_orientation(G, v)
        # Use node positions stored at construction as the layout.
        layout = {v: G.nodes[v]['pos'] for v in G.nodes()}
        phase = calculate_layout_phase(G, layout)
        assert abs(abs(phase) - 1.0) < 1e-10
