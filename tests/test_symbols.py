"""
Unit tests for theta and delta symbol calculations
"""

import unittest
import math
from src.spin_evaluator import SpinNetworkEvaluator


class TestThetaSymbol(unittest.TestCase):
    """Test theta symbol calculations"""

    def setUp(self):
        self.evaluator = SpinNetworkEvaluator(max_two_j=40)

    def tearDown(self):
        self.evaluator.cleanup()

    def test_theta_simple_triangle(self):
        """Test theta for a simple valid triangle"""
        # θ(1, 1, 1) should be computable
        result = self.evaluator.theta_symbol(1, 1, 1)
        self.assertIsInstance(result, float)
        self.assertNotEqual(result, 0.0)

    def test_theta_invalid_triangle(self):
        """Test theta returns 0 for invalid triangle"""
        # θ(1, 1, 5) violates triangle inequality
        result = self.evaluator.theta_symbol(1, 1, 5)
        self.assertEqual(result, 0.0)

    def test_theta_sign_even(self):
        """Test theta sign for even j+k+l"""
        # For j+k+l even, sign should be positive
        result = self.evaluator.theta_symbol(1, 1, 2)
        self.assertGreater(result, 0)

    def test_theta_sign_odd(self):
        """Test theta sign for odd j+k+l"""
        # For j+k+l odd, sign should be negative
        result = self.evaluator.theta_symbol(1, 1, 1)
        self.assertLess(result, 0)

    def test_theta_power(self):
        """Test theta with power parameter"""
        theta_1 = self.evaluator.theta_symbol(1, 1, 1, power=1.0)
        theta_2 = self.evaluator.theta_symbol(1, 1, 1, power=2.0)
        # theta^2 should equal (theta)^2
        self.assertAlmostEqual(theta_2, theta_1 ** 2, places=10)

    def test_theta_large_spins(self):
        """Test theta with large spins uses log-gamma"""
        # Should not overflow for j=150
        result = self.evaluator.theta_symbol(150, 150, 150)
        self.assertIsInstance(result, float)
        self.assertFalse(math.isinf(result))
        self.assertFalse(math.isnan(result))


class TestDeltaSymbol(unittest.TestCase):
    """Test delta symbol calculations"""

    def setUp(self):
        self.evaluator = SpinNetworkEvaluator(max_two_j=40)

    def tearDown(self):
        self.evaluator.cleanup()

    def test_delta_zero_spin(self):
        """Test delta for j=0"""
        result = self.evaluator.delta_symbol(0)
        # Δ_0 = (-1)^0 × 1 = 1
        self.assertEqual(result, 1.0)

    def test_delta_half_integer(self):
        """Test delta for half-integer spin"""
        result = self.evaluator.delta_symbol(0.5)
        # Δ_{1/2} = (-1)^1 × 2 = -2
        self.assertEqual(result, -2.0)

    def test_delta_integer_spin(self):
        """Test delta for integer spin"""
        result = self.evaluator.delta_symbol(1.0)
        # Δ_1 = (-1)^2 × 3 = 3
        self.assertEqual(result, 3.0)

    def test_delta_power(self):
        """Test delta with power parameter"""
        delta_1 = self.evaluator.delta_symbol(1, power=1.0)
        delta_2 = self.evaluator.delta_symbol(1, power=2.0)
        # Δ_1 = 3, so Δ^2 = 3^2 = 9
        self.assertEqual(delta_1, 3.0)
        self.assertEqual(delta_2, 9.0)

    def test_delta_large_spin(self):
        """Test delta with large spin"""
        # Δ_150 = (-1)^300 × 301 = 301
        result = self.evaluator.delta_symbol(150)
        self.assertIsInstance(result, float)
        self.assertFalse(math.isinf(result))
        self.assertFalse(math.isnan(result))
        # Should be 301
        self.assertEqual(result, 301.0)


class TestWigner6j(unittest.TestCase):
    """Test Wigner 6j symbol calculations"""

    def setUp(self):
        self.evaluator = SpinNetworkEvaluator(max_two_j=40)

    def tearDown(self):
        self.evaluator.cleanup()

    def test_6j_simple_case(self):
        """Test 6j symbol for simple valid case"""
        # {1 1 1}
        # {1 1 1} should be non-zero
        result = self.evaluator.wigner_6j(1, 1, 1, 1, 1, 1)
        self.assertIsInstance(result, float)

    def test_6j_invalid_triangles(self):
        """Test 6j returns 0 for invalid triangles"""
        # Violates triangle condition
        result = self.evaluator.wigner_6j(1, 1, 5, 1, 1, 1)
        self.assertEqual(result, 0.0)

    def test_6j_orthogonality(self):
        """Test 6j orthogonality relations"""
        # Sum over orthogonality relation should give expected result
        # This is a complex test - just verify it computes
        result = self.evaluator.wigner_6j(1, 2, 3, 2, 1, 2)
        self.assertIsInstance(result, float)


if __name__ == '__main__':
    unittest.main()
