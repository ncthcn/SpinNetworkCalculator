"""
Integration tests for complete spin network evaluation pipeline
"""

import unittest
import os
import networkx as nx
from src.graph_reducer import reduce_all_cycles
from src.spin_evaluator import SpinNetworkEvaluator


class TestEndToEndEvaluation(unittest.TestCase):
    """Test complete pipeline from graph to numerical result"""

    def setUp(self):
        """Set up test graphs and evaluator"""
        self.evaluator = SpinNetworkEvaluator(max_two_j=40)

    def tearDown(self):
        """Clean up evaluator"""
        self.evaluator.cleanup()

    def test_simple_triangle_evaluation(self):
        """Test evaluating a simple triangle spin network"""
        # Create a simple triangle
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 0, label=1)

        # Reduce
        reduced_terms = reduce_all_cycles(G)
        self.assertGreater(len(reduced_terms), 0)

        # Should complete without errors
        self.assertIsInstance(reduced_terms, list)

    def test_mixed_spin_values(self):
        """Test graph with different spin values"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=2)
        G.add_edge(2, 0, label=2)

        # Reduce
        reduced_terms = reduce_all_cycles(G)
        self.assertGreater(len(reduced_terms), 0)


class TestSymbolicVariables(unittest.TestCase):
    """Test handling of symbolic F-variables"""

    def setUp(self):
        self.evaluator = SpinNetworkEvaluator(max_two_j=40)

    def tearDown(self):
        self.evaluator.cleanup()

    def test_graph_with_f_variables(self):
        """Test expression with F-variables (from F-moves)"""
        # Create a graph that will generate F-variables
        G = nx.MultiGraph()
        # Square that will be F-moved
        G.add_edge(0, 1, label=1)
        G.add_edge(1, 2, label=1)
        G.add_edge(2, 3, label=1)
        G.add_edge(3, 0, label=1)

        reduced_terms = reduce_all_cycles(G)

        # Should complete without errors
        self.assertGreater(len(reduced_terms), 0)


class TestLargeSpins(unittest.TestCase):
    """Test handling of large spin values"""

    def setUp(self):
        # Use larger max_two_j for these tests
        self.evaluator = SpinNetworkEvaluator(max_two_j=200)

    def tearDown(self):
        self.evaluator.cleanup()

    def test_large_spin_triangle(self):
        """Test triangle with large spin values"""
        G = nx.MultiGraph()
        G.add_edge(0, 1, label=10)
        G.add_edge(1, 2, label=10)
        G.add_edge(2, 0, label=10)

        reduced_terms = reduce_all_cycles(G)
        self.assertGreater(len(reduced_terms), 0)

    def test_very_large_spin_theta(self):
        """Test theta symbol with very large spins"""
        # Test theta symbol directly
        result = self.evaluator.theta_symbol(100, 100, 100)

        import math
        self.assertIsInstance(result, float)
        self.assertFalse(math.isinf(result))
        self.assertFalse(math.isnan(result))

    def test_very_large_spin_delta(self):
        """Test delta symbol with very large spins"""
        result = self.evaluator.delta_symbol(100)

        import math
        self.assertIsInstance(result, float)
        self.assertGreater(result, 0)
        # Should be finite
        self.assertFalse(math.isinf(result))


if __name__ == '__main__':
    unittest.main()
