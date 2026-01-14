"""
Unit tests for F-variable range computation
"""

import unittest
from src.utils import f_range_symbolic, f_range_with_symbolic, is_numeric_label


class TestFRangeSymbolic(unittest.TestCase):
    """Test F-range computation for numeric labels"""

    def test_numeric_simple_case(self):
        """Test F-range for simple numeric case"""
        # F-range for edges (1, 1, 1, 1)
        rng = f_range_symbolic(1, 1, 1, 1)
        self.assertIsNotNone(rng)
        self.assertIn('Fmin', rng)
        self.assertIn('Fmax', rng)
        # F must satisfy: |1-1| ≤ F ≤ 1+1 and |F-1| ≤ 1 ≤ F+1
        # So: 0 ≤ F ≤ 2
        self.assertEqual(rng['Fmin'], 0)
        self.assertEqual(rng['Fmax'], 4)  # 2*2 in 2j notation

    def test_numeric_half_integer(self):
        """Test F-range for half-integer spins"""
        rng = f_range_symbolic(0.5, 0.5, 0.5, 0.5)
        self.assertIsNotNone(rng)
        # Should handle half-integers correctly
        self.assertIsInstance(rng['Fmin'], int)
        self.assertIsInstance(rng['Fmax'], int)

    def test_numeric_parity(self):
        """Test parity computation"""
        rng = f_range_symbolic(1, 1, 1, 1)
        self.assertIsNotNone(rng)
        self.assertIn('parity', rng)
        # Parity should be 0 or 1
        self.assertIn(rng['parity'], [0, 1])


class TestFRangeWithSymbolic(unittest.TestCase):
    """Test F-range computation with symbolic labels"""

    def test_all_numeric(self):
        """Test with all numeric labels"""
        rng = f_range_with_symbolic(1, 1, 1, 1)
        self.assertIsNotNone(rng)
        self.assertFalse(rng.get('symbolic', True))
        # Should use exact range
        self.assertIn('Fmin', rng)
        self.assertIn('Fmax', rng)

    def test_one_symbolic(self):
        """Test with one symbolic label"""
        rng = f_range_with_symbolic("F_1", 1, 1, 1)
        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))
        # Should compute tighter range based on numeric labels
        self.assertEqual(rng['Fmin'], 0)
        # With improved algorithm, Fmax is computed from numeric labels
        self.assertLessEqual(rng['Fmax'], 40)  # At most conservative

    def test_all_symbolic(self):
        """Test with all symbolic labels"""
        rng = f_range_with_symbolic("F_1", "F_2", "F_3", "F_4")
        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))
        # Should use range based on default bounds for unknowns
        self.assertEqual(rng['Fmin'], 0)
        # With all symbolic, uses conservative default [0,40] for each var
        # So Fmax can be up to 40+40=80 from the sum constraint
        self.assertLessEqual(rng['Fmax'], 80)

    def test_symbolic_dependencies(self):
        """Test tracking of symbolic dependencies"""
        rng = f_range_with_symbolic("F_1", "F_2", 1, 1)
        self.assertIsNotNone(rng)
        self.assertIn('depends_on', rng)
        # Should track which variables it depends on
        depends = rng['depends_on']
        self.assertIn("F_1", depends)
        self.assertIn("F_2", depends)


class TestIsNumericLabel(unittest.TestCase):
    """Test numeric label detection"""

    def test_integer(self):
        """Test integer detection"""
        self.assertTrue(is_numeric_label(1))
        self.assertTrue(is_numeric_label(0))
        self.assertTrue(is_numeric_label(10))

    def test_float(self):
        """Test float detection"""
        self.assertTrue(is_numeric_label(1.0))
        self.assertTrue(is_numeric_label(0.5))
        self.assertTrue(is_numeric_label(1.5))

    def test_string_number(self):
        """Test string number detection"""
        self.assertTrue(is_numeric_label("1"))
        self.assertTrue(is_numeric_label("0.5"))
        self.assertTrue(is_numeric_label("1.5"))

    def test_symbolic_string(self):
        """Test symbolic string detection"""
        self.assertFalse(is_numeric_label("F_1"))
        self.assertFalse(is_numeric_label("F_2"))
        self.assertFalse(is_numeric_label("variable"))


if __name__ == '__main__':
    unittest.main()
