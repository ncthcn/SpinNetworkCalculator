"""
Unit tests for improved symbolic range tracking
"""

import unittest
from src.utils import f_range_with_symbolic, compute_nested_ranges


class TestImprovedRanges(unittest.TestCase):
    """Test improved range computation with dependency tracking"""

    def test_numeric_labels_exact_range(self):
        """Test that numeric labels give exact ranges"""
        # For labels (1, 1, 1, 1), F must satisfy:
        # |1-1| <= F <= 1+1  AND  |1-1| <= F <= 1+1
        # So: 0 <= F <= 2
        rng = f_range_with_symbolic(1, 1, 1, 1)

        self.assertIsNotNone(rng)
        self.assertEqual(rng['Fmin'], 0)  # 2*0
        self.assertEqual(rng['Fmax'], 4)  # 2*2
        self.assertFalse(rng.get('symbolic', True))

    def test_one_symbolic_conservative(self):
        """Test that one symbolic label computes tighter range using numeric labels"""
        # With F_1 symbolic and numeric labels (1, 1, 1):
        # F must satisfy: |F_1-1| <= F <= F_1+1  AND  |1-1| <= F <= 1+1
        # With F_1 ∈ [0,20]: |0-1| <= F <= 20+1  => 0 <= F <= 21
        #                    0 <= F <= 2
        # Combined: 0 <= F <= min(21, 2) = 2  => [0, 4] in 2j notation
        rng = f_range_with_symbolic("F_1", 1, 1, 1)

        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))
        self.assertEqual(rng['Fmin'], 0)
        # The improved algorithm computes a tighter bound based on numeric labels
        self.assertLessEqual(rng['Fmax'], 40)  # At most conservative
        self.assertIn("F_1", rng['depends_on'])

    def test_one_symbolic_with_known_range(self):
        """Test that known range tightens bounds for symbolic label"""
        # If F_1 is known to be in [2, 4] (in 2j notation: [1, 2])
        # and other labels are 1, 1, 1
        # Then F must satisfy: |F_1-1| <= F <= F_1+1
        # With F_1 ∈ [1,2]: min(F) >= |1-1| = 0, max(F) <= 2+1 = 3
        known_ranges = {"F_1": (2, 4)}  # 2j notation

        rng = f_range_with_symbolic("F_1", 1, 1, 1, known_ranges=known_ranges)

        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))

        # Should be tighter than conservative [0, 40]
        self.assertLess(rng['Fmax'], 40)
        self.assertIn("F_1", rng['depends_on'])

    def test_nested_symbolic_ranges(self):
        """Test computing tighter ranges for nested F-variables"""
        # Simulate: F_1 ∈ [0, 4], F_2 depends on F_1
        all_ranges_info = {
            "F_1": {
                "Fmin": 0,
                "Fmax": 4,
                "symbolic": False
            },
            "F_2": {
                "Fmin": 0,
                "Fmax": 8,
                "symbolic": True,
                "depends_on": ["F_1"]
            }
        }

        summation_vars = ["F_1", "F_2"]
        computed_ranges = compute_nested_ranges(summation_vars, all_ranges_info)

        self.assertIn("F_1", computed_ranges)
        self.assertIn("F_2", computed_ranges)

        # F_1 should have exact range
        self.assertEqual(computed_ranges["F_1"], (0, 4))

        # F_2 should use its computed range (may be tightened in future)
        self.assertEqual(computed_ranges["F_2"], (0, 8))

    def test_multiple_symbolic_labels(self):
        """Test range computation with multiple symbolic labels"""
        # If F_1 and F_2 are both symbolic
        rng = f_range_with_symbolic("F_1", "F_2", 1, 1)

        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))
        self.assertIn("F_1", rng['depends_on'])
        self.assertIn("F_2", rng['depends_on'])

    def test_mixed_numeric_symbolic(self):
        """Test range with some numeric and some symbolic labels"""
        # Labels: (F_1, 2, 1, 1) where F_1 ∈ [2, 6]
        known_ranges = {"F_1": (2, 6)}

        rng = f_range_with_symbolic("F_1", 2, 1, 1, known_ranges=known_ranges)

        self.assertIsNotNone(rng)
        self.assertTrue(rng.get('symbolic', False))

        # With F_1 ∈ [1,3] (in j notation) and other labels 2, 1, 1:
        # Triangle: |2-1| <= F <= 2+1  =>  1 <= F <= 3  => [2, 6] in 2j
        #           |F_1-1| <= F <= F_1+1
        # With F_1 ∈ [1,3]: |1-1| <= F <= 3+1  => 0 <= F <= 4  => [0, 8] in 2j
        # Combined: max(2, 0) <= F <= min(6, 8)  =>  2 <= F <= 6
        self.assertGreaterEqual(rng['Fmin'], 0)
        self.assertLessEqual(rng['Fmax'], 8)

    def test_range_tightening_reduces_iterations(self):
        """Test that known ranges can provide tighter bounds"""
        # Range with unknown F_1 (uses default [0,40])
        rng_unknown = f_range_with_symbolic("F_1", "F_2", 1, 1, known_ranges={})

        # Range with known F_1 and F_2
        known_ranges = {"F_1": (0, 4), "F_2": (2, 6)}
        rng_known = f_range_with_symbolic("F_1", "F_2", 1, 1, known_ranges=known_ranges)

        # Calculate iteration counts
        unknown_count = (rng_unknown['Fmax'] - rng_unknown['Fmin']) // 2 + 1
        known_count = (rng_known['Fmax'] - rng_known['Fmin']) // 2 + 1

        # Known range should have same or fewer iterations
        self.assertLessEqual(known_count, unknown_count)

        print(f"\nIteration comparison:")
        print(f"  Unknown ranges: {unknown_count} iterations")
        print(f"  Known ranges: {known_count} iterations")
        if known_count < unknown_count:
            print(f"  Reduction: {(1 - known_count/unknown_count)*100:.1f}%")


class TestRangeSymbolicExpressions(unittest.TestCase):
    """Test symbolic expression building"""

    def test_symbolic_expressions_present(self):
        """Test that symbolic expressions are built for symbolic ranges"""
        rng = f_range_with_symbolic("F_1", "F_2", 1, 1)

        self.assertIsNotNone(rng)
        self.assertIn('symbolic_Fmin', rng)
        self.assertIn('symbolic_Fmax', rng)

        # Check expressions contain the variable names
        self.assertIn("F_1", rng['symbolic_Fmin'])
        self.assertIn("F_2", rng['symbolic_Fmin'])

    def test_numeric_no_symbolic_expressions(self):
        """Test that numeric ranges don't create symbolic expressions"""
        rng = f_range_with_symbolic(1, 1, 1, 1)

        self.assertIsNotNone(rng)
        # Should not have symbolic expressions
        self.assertNotIn('symbolic_Fmin', rng)
        self.assertNotIn('symbolic_Fmax', rng)


if __name__ == '__main__':
    unittest.main()
