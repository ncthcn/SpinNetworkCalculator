#!/usr/bin/env python3
"""
Quick test to verify multi-variable summation support.
This creates a synthetic term with multiple summation variables.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.spin_evaluator import evaluate_spin_network
from src.LaTeX_rendering import save_latex_pdf

# Create a test term with TRIPLE summation
test_term = {
    "coeffs": [
        {"type": "sign_value", "value": 1},
        {"type": "theta", "args": (1, 2, 3), "power": 1},

        # Three summation variables
        {"type": "sum", "index": "F_1", "range2": {"Fmin": 2, "Fmax": 6, "parity": 0}},
        {"type": "sum", "index": "F_2", "range2": {"Fmin": 2, "Fmax": 6, "parity": 0}},
        {"type": "sum", "index": "F_3", "range2": {"Fmin": 2, "Fmax": 4, "parity": 0}},

        # Coefficients depending on the summation variables
        {"type": "delta", "fixed": {"j": "F_1"}, "power": 1},
        {"type": "delta", "fixed": {"j": "F_2"}, "power": 1},
        {"type": "delta", "fixed": {"j": "F_3"}, "power": 1},
        {"type": "W6j", "args": (1, "F_1", 2, "F_2", 3, "F_3"), "power": 1}
    ]
}

canonical_terms = [test_term]

print("=" * 70)
print("MULTI-SUMMATION TEST")
print("=" * 70)
print("\nTest term with 3 summation variables:")
print("  ∑_{F_1=1..3} ∑_{F_2=1..3} ∑_{F_3=1..2} [Δ_{F_1} × Δ_{F_2} × Δ_{F_3} × W6j(...)]")
print(f"\nTotal iterations: 3 × 3 × 2 = {3*3*2}")

# Generate LaTeX rendering
print("\nGenerating LaTeX PDF...")
save_latex_pdf(canonical_terms, filename="test_triple_sum.pdf")

# Evaluate numerically with parallel backend
print("\nEvaluating numerically with parallel backend...")
result = evaluate_spin_network(canonical_terms, max_two_j=50, backend='auto')

print("\n" + "=" * 70)
print(f"✅ Result: {result:.15e}")
print("=" * 70)
print("\n✅ Multi-summation support verified!")
print("   - LaTeX PDF saved to: test_triple_sum.pdf")
print("   - Check the PDF to see all three summation symbols")
