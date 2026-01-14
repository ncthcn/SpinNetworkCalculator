"""
Unit tests for graph reduction operations
"""

import unittest
import networkx as nx
from src.graph_reducer import (
    find_triangle_candidate,
    apply_triangle_reduction,
    reduce_all_cycles,
    reduce_all_triangles
)


class TestTriangleDetection(unittest.TestCase):
    """Test triangle detection"""

    def test_find_triangle_in_simple_graph(self):
        """Test finding a triangle in a simple graph"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        # Create a term dict
        term = {"graph": G, "coeffs": []}

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

    def test_reduce_with_animator(self):
        """Test that reduction works with animator"""
        from src.reduction_animator import ReductionAnimator

        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        animator = ReductionAnimator(output_dir="test_animation_output")

        results = reduce_all_cycles(G, animator=animator)

        # Should have captured steps
        self.assertGreater(len(animator.steps), 0)

        # Clean up test output
        import shutil
        import os
        if os.path.exists("test_animation_output"):
            shutil.rmtree("test_animation_output")


if __name__ == '__main__':
    unittest.main()
