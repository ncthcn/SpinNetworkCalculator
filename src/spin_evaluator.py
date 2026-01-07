"""
Spin Network Expression Evaluator

This module evaluates canonical spin network expressions numerically using wigxjpf.

EDUCATIONAL OVERVIEW:
====================

1. WHAT WE'RE COMPUTING:
   The norm of a spin network, which is a product of:
   - Wigner 6j symbols (quantum recoupling coefficients)
   - Theta symbols θ(j1,j2,j3) = √[(2j1+1)(2j2+1)(2j3+1)]
   - Delta symbols Δ_j = √(2j+1)
   - Sign factors (-1)^(...)
   - Summations over intermediate quantum numbers

2. WHY WIGXJPF:
   - Uses prime factorization to avoid overflow
   - Handles half-integer spins (stored as 2*j)
   - Industry standard, extensively tested
   - Much faster than naive implementations

3. OPTIMIZATION STRATEGIES:
   - Pre-initialize tables for maximum j value
   - Cache repeated calculations
   - Vectorize summations where possible
   - Reuse temporary arrays

4. INPUT FORMAT:
   Canonical terms from norm_reducer.py:
   [
       {
           "coeffs": [
               {"type": "sign_value", "value": -1},
               {"type": "theta", "args": (j1, j2, j3), "power": p},
               {"type": "sum", "index": "F_1", "range2": {...}},
               {"type": "delta", "fixed": {"j": "F_1"}, "power": p},
               {"type": "W6j", "args": (...), "power": p},
               ...
           ]
       }
   ]
"""

import pywigxjpf as wig
import numpy as np
from typing import List, Dict, Any, Tuple
import math


class SpinNetworkEvaluator:
    """
    Evaluates canonical spin network expressions using wigxjpf.

    USAGE:
        evaluator = SpinNetworkEvaluator(max_two_j=200)
        result = evaluator.evaluate(canonical_terms)
        evaluator.cleanup()
    """

    def __init__(self, max_two_j: int = 200):
        """
        Initialize the evaluator with wigxjpf tables.

        Parameters:
        -----------
        max_two_j : int
            Maximum value of 2*j that will be used. Default 200 (j up to 100).

            WHY 2*j?: In quantum mechanics, angular momentum can be half-integer
            (e.g., j = 1/2, 3/2, 5/2...). We store as 2*j to use integers:
            - j = 1/2  → 2*j = 1
            - j = 1    → 2*j = 2
            - j = 3/2  → 2*j = 3
            etc.

        MEMORY USAGE:
            The tables scale as O(max_two_j^2), so be mindful for large j.
            max_two_j=200 uses ~few MB.
            max_two_j=1000 uses ~hundreds of MB.
        """
        self.max_two_j = max_two_j
        self.initialized = False

        print(f"Initializing wigxjpf tables for max 2j = {max_two_j}...")

        # Initialize for 6j symbols (wigner_type=6)
        wig.wig_table_init(max_two_j, 6)

        # Allocate temporary calculation arrays
        # These are reused across calculations for efficiency
        wig.wig_temp_init(max_two_j)

        self.initialized = True
        print("✓ Wigxjpf initialized and ready")

    def cleanup(self):
        """
        Free wigxjpf memory. Call this when done with all calculations.

        IMPORTANT: Always call this to avoid memory leaks!
        """
        if self.initialized:
            wig.wig_temp_free()
            wig.wig_table_free()
            self.initialized = False
            print("✓ Wigxjpf memory freed")

    def __del__(self):
        """Destructor - cleanup if user forgets"""
        self.cleanup()

    @staticmethod
    def convert_to_two_j(value: Any) -> int:
        """
        Convert a spin value to 2*j representation.

        Parameters:
        -----------
        value : float, int, or str
            The spin value, could be:
            - Float: 1.5 → 3 (since 2*1.5 = 3)
            - Int: 2 → 4 (since 2*2 = 4)
            - Str: "F_1" → keep as variable name for later substitution

        Returns:
        --------
        int or str : The 2*j value (or variable name if symbolic)

        EXAMPLES:
            convert_to_two_j(0.5) → 1   (j=1/2)
            convert_to_two_j(1.0) → 2   (j=1)
            convert_to_two_j(2.5) → 5   (j=5/2)
            convert_to_two_j("F_1") → "F_1"  (variable)
        """
        if isinstance(value, str):
            return value  # Keep variables as strings

        # Convert numeric value to 2*j
        two_j = int(round(2 * float(value)))
        return two_j

    def theta_symbol(self, j1, j2, j3, power=1.0):
        """
        Compute θ(j1,j2,j3)^power = [(2j1+1)(2j2+1)(2j3+1)]^(power/2)

        PHYSICS: Theta symbol encodes the quantum dimensions of the coupling.
        It appears when normalizing Clebsch-Gordan coefficients.

        Parameters:
        -----------
        j1, j2, j3 : numeric values (will be converted to 2*j)
        power : float, exponent on the theta symbol

        Returns:
        --------
        float : The numerical value

        FORMULA:
            θ(j1,j2,j3) = √[(2j1+1)(2j2+1)(2j3+1)]
            θ(j1,j2,j3)^p = [(2j1+1)(2j2+1)(2j3+1)]^(p/2)
        """
        two_j1 = self.convert_to_two_j(j1)
        two_j2 = self.convert_to_two_j(j2)
        two_j3 = self.convert_to_two_j(j3)

        # Calculate dimensions: d_j = 2j + 1
        d1 = two_j1 + 1
        d2 = two_j2 + 1
        d3 = two_j3 + 1

        # θ^p = (d1 * d2 * d3)^(p/2)
        result = math.pow(d1 * d2 * d3, power / 2.0)
        return result

    def delta_symbol(self, j, power=1.0):
        """
        Compute Δ_j^power = (2j+1)^(power/2)

        PHYSICS: Quantum dimension of spin-j representation.
        Appears in trace normalizations.

        Parameters:
        -----------
        j : numeric value or variable
        power : float, exponent

        Returns:
        --------
        float : The numerical value

        FORMULA:
            Δ_j = √(2j+1)
            Δ_j^p = (2j+1)^(p/2)
        """
        two_j = self.convert_to_two_j(j)
        dimension = two_j + 1
        result = math.pow(dimension, power / 2.0)
        return result

    def wigner_6j(self, j1, j2, j3, j4, j5, j6, power=1.0):
        """
        Compute Wigner 6j symbol using wigxjpf.

        PHYSICS: The 6j symbol represents the transformation coefficient
        between two different coupling schemes of three angular momenta.

        NOTATION: {j1 j2 j3}
                  {j4 j5 j6}

        TRIANGLE CONDITIONS: Must satisfy 4 triangle inequalities:
        - (j1, j2, j3) form a triangle
        - (j4, j5, j3) form a triangle
        - (j1, j5, j6) form a triangle
        - (j4, j2, j6) form a triangle

        Parameters:
        -----------
        j1-j6 : numeric spin values
        power : float, exponent on the 6j symbol

        Returns:
        --------
        float : The numerical value (0 if triangle conditions fail)

        COMPUTATIONAL NOTE:
            wigxjpf returns 0 automatically if triangle conditions aren't met.
            No need to check explicitly.
        """
        two_j1 = self.convert_to_two_j(j1)
        two_j2 = self.convert_to_two_j(j2)
        two_j3 = self.convert_to_two_j(j3)
        two_j4 = self.convert_to_two_j(j4)
        two_j5 = self.convert_to_two_j(j5)
        two_j6 = self.convert_to_two_j(j6)

        # Call wigxjpf's 6j function
        # Arguments are all 2*j values (integers)
        value = wig.wig6jj(two_j1, two_j2, two_j3, two_j4, two_j5, two_j6)

        # Apply power if needed
        if power != 1.0:
            value = math.pow(value, power)

        return value

    def evaluate_term(self, term: Dict[str, Any]) -> float:
        """
        Evaluate a single canonical term.

        A term is a product of coefficients, some of which may be inside summations.

        Parameters:
        -----------
        term : dict with key "coeffs" containing list of coefficient dicts

        Returns:
        --------
        float : The numerical value of the term

        ALGORITHM:
        1. Extract summation variables and their ranges
        2. Separate coefficients into "before sum" and "inside sum"
        3. Compute pre-factors (constants, independent coefficients)
        4. Loop over summation indices
        5. For each sum value, compute product of dependent coefficients
        6. Accumulate the sum
        7. Multiply by pre-factors

        EXAMPLE STRUCTURE:
            -1 × θ(1,2,3) × θ(4,5,6) × ∑_{F=2..5} [Δ_F × {1 2 F; 3 4 5}_W]

            Pre-factors: -1 × θ(1,2,3) × θ(4,5,6)
            Sum over F from 2 to 5 of: Δ_F × {1 2 F; 3 4 5}
        """
        coeffs = term.get("coeffs", [])

        # Step 1: Extract summation information
        sum_vars = {}  # {variable_name: (min_value, max_value)}

        for c in coeffs:
            if isinstance(c, dict) and c.get("type") == "sum":
                var = c.get("index", "f")
                range_info = c.get("range2", {})

                # Extract min and max (stored as 2*j)
                min_two_j = range_info.get("Fmin", 0)
                max_two_j = range_info.get("Fmax", 0)

                # Convert back to j values
                min_j = min_two_j // 2
                max_j = max_two_j // 2

                sum_vars[var] = (min_j, max_j)
                print(f"  Summation: {var} from {min_j} to {max_j}")

        # Step 2: Evaluate constant pre-factors (coefficients not in sum)
        pre_factor = 1.0

        for c in coeffs:
            if not isinstance(c, dict):
                continue

            typ = c.get("type")

            # Skip summations (already processed)
            if typ == "sum":
                continue

            # Check if this coefficient depends on sum variables
            depends_on_sum = self._depends_on_sum_var(c, sum_vars.keys())

            if not depends_on_sum:
                # Evaluate immediately and add to pre-factor
                val = self._evaluate_coefficient(c, {})
                pre_factor *= val

        # Step 3: If no summations, we're done
        if not sum_vars:
            return pre_factor

        # Step 4: Perform summation
        sum_result = 0.0

        # Generate all combinations of sum indices
        # For now, assume single summation (can extend to multiple)
        if len(sum_vars) == 1:
            var_name, (min_val, max_val) = list(sum_vars.items())[0]

            for sum_value in range(min_val, max_val + 1):
                # Create substitution dict
                substitutions = {var_name: sum_value}

                # Evaluate all coefficients that depend on sum variable
                term_value = 1.0

                for c in coeffs:
                    if not isinstance(c, dict):
                        continue

                    typ = c.get("type")
                    if typ == "sum":
                        continue

                    depends_on_sum = self._depends_on_sum_var(c, sum_vars.keys())

                    if depends_on_sum:
                        val = self._evaluate_coefficient(c, substitutions)
                        term_value *= val

                sum_result += term_value

        else:
            raise NotImplementedError("Multiple summations not yet implemented")

        # Step 5: Combine pre-factor and sum
        total = pre_factor * sum_result
        return total

    def _depends_on_sum_var(self, coeff: Dict, sum_var_names: set) -> bool:
        """Check if a coefficient depends on any summation variable."""
        # Check in args
        args = coeff.get("args", ())
        for arg in args:
            if isinstance(arg, str) and arg in sum_var_names:
                return True

        # Check in fixed dict
        fixed = coeff.get("fixed", {})
        for val in fixed.values():
            if isinstance(val, str) and val in sum_var_names:
                return True

        return False

    def _evaluate_coefficient(self, coeff: Dict, substitutions: Dict[str, int]) -> float:
        """
        Evaluate a single coefficient with variable substitutions.

        Parameters:
        -----------
        coeff : dict describing the coefficient
        substitutions : dict mapping variable names to numeric values

        Returns:
        --------
        float : The numerical value
        """
        typ = coeff.get("type")

        if typ == "sign_value":
            return float(coeff.get("value", 1))

        elif typ == "theta":
            args = coeff.get("args", ())
            power = coeff.get("power", 1)

            # Substitute variables
            j1, j2, j3 = args
            j1 = substitutions.get(j1, j1) if isinstance(j1, str) else j1
            j2 = substitutions.get(j2, j2) if isinstance(j2, str) else j2
            j3 = substitutions.get(j3, j3) if isinstance(j3, str) else j3

            return self.theta_symbol(j1, j2, j3, power)

        elif typ == "delta":
            fixed = coeff.get("fixed", {})
            j = fixed.get("j")
            power = coeff.get("power", 1)

            # Substitute variable
            j = substitutions.get(j, j) if isinstance(j, str) else j

            return self.delta_symbol(j, power)

        elif typ == "W6j":
            args = coeff.get("args", ())
            power = coeff.get("power", 1)

            # Substitute variables
            j_vals = []
            for j in args:
                j_val = substitutions.get(j, j) if isinstance(j, str) else j
                j_vals.append(j_val)

            return self.wigner_6j(*j_vals, power=power)

        else:
            print(f"Warning: Unknown coefficient type '{typ}'")
            return 1.0

    def evaluate(self, canonical_terms: List[Dict]) -> float:
        """
        Evaluate a list of canonical terms (usually just one term).

        Parameters:
        -----------
        canonical_terms : list of term dicts from canonicalise_terms()

        Returns:
        --------
        float : The numerical value of the spin network norm

        USAGE:
            from norm_reducer import canonicalise_terms
            canon_terms = canonicalise_terms(clean_terms)
            result = evaluator.evaluate(canon_terms)
        """
        if not self.initialized:
            raise RuntimeError("Evaluator not initialized. Create new instance.")

        print("\n" + "=" * 70)
        print("EVALUATING SPIN NETWORK EXPRESSION")
        print("=" * 70)

        total_result = 0.0

        for i, term in enumerate(canonical_terms):
            print(f"\nEvaluating term {i + 1}/{len(canonical_terms)}...")
            term_value = self.evaluate_term(term)
            print(f"  Term value: {term_value:.10e}")
            total_result += term_value

        print("\n" + "=" * 70)
        print(f"FINAL RESULT: {total_result:.15e}")
        print("=" * 70)

        return total_result


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def evaluate_spin_network(canonical_terms: List[Dict], max_two_j: int = 200) -> float:
    """
    Convenience function to evaluate spin network in one call.

    Parameters:
    -----------
    canonical_terms : output from canonicalise_terms()
    max_two_j : maximum 2*j value expected

    Returns:
    --------
    float : numerical result

    EXAMPLE:
        result = evaluate_spin_network(canon_terms, max_two_j=100)
    """
    evaluator = SpinNetworkEvaluator(max_two_j)
    try:
        result = evaluator.evaluate(canonical_terms)
        return result
    finally:
        evaluator.cleanup()
