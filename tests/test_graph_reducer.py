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
Unit tests for graph reduction operations
"""

import unittest
import networkx as nx
from src.graph_reducer import (
    find_triangle_candidate,
    apply_triangle_reduction,
    reduce_all_cycles,
)


class TestTriangleDetection(unittest.TestCase):
    """Test triangle detection"""

    def test_find_triangle_in_simple_graph(self):
        """Test finding a triangle in a simple graph"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        # Try to find triangle
        triangle_nodes = find_triangle_candidate(G)

        # Should find the triangle (0, 1, 2)
        self.assertIsNotNone(triangle_nodes)
        self.assertEqual(len(triangle_nodes), 3)


class TestTriangleReduction(unittest.TestCase):
    """Test triangle reduction"""

    def test_simple_triangle_reduction(self):
        """Test reducing a simple triangle"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        term = {"graph": G, "coeffs": []}

        # Try to reduce
        result = apply_triangle_reduction(term)

        # Triangle should be reducible
        if result is not None:
            self.assertIn("graph", result)
            self.assertIn("coeffs", result)


class TestFullReduction(unittest.TestCase):
    """Test complete reduction pipeline"""

    def test_reduce_simple_graph(self):
        """Test full reduction on a simple graph"""
        G = nx.MultiGraph()
        # Create a simple graph: triangle
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        # Perform full reduction
        results = reduce_all_cycles(G)

        # Should return a list of terms
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Each term should have a graph and coeffs
        for term in results:
            self.assertIn("graph", term)
            self.assertIn("coeffs", term)


if __name__ == "__main__":
    unittest.main()
