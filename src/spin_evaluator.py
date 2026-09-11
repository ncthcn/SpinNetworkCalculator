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
Spin Network Expression Evaluator

This module evaluates canonical spin network expressions numerically using wigxjpf.

EDUCATIONAL OVERVIEW:
====================

1. WHAT WE'RE COMPUTING:
   The norm of a spin network, which is a product of:
   - Wigner 6j symbols (SU(2) recoupling coefficients)
   - Theta symbols
         θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
   - Delta symbols
         Δ_j = (-1)^(2j) × (2j+1)
   - Sign factors (-1)^(...)
   - Summations over intermediate spins

   SIGNS: every factor above carries its own sign and keeps it throughout the
   computation -- the signs cancel against one another, so discarding them
   early gives the wrong magnitude. The modulus is a property of the *norm*,
   not of the individual factors, and is therefore applied exactly once at the
   very end, by Formula.evaluate_numeric() in src/api.py, immediately before
   the norm is returned to the caller.

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
from typing import List, Dict, Any, Tuple, Optional
import ast
import math
import itertools
from multiprocessing import Pool, cpu_count
import os
import sys
import time
from functools import lru_cache

# Cached factorial for performance optimization
@lru_cache(maxsize=512)
def cached_factorial(n: int) -> int:
    """
    Cached factorial computation for performance.

    For spin networks, j values are typically small (0 to ~20),
    so factorials up to ~60! are needed. Caching gives massive speedup.
    """
    return math.factorial(n)


class SpinNetworkEvaluator:
    """
    Evaluates canonical spin network expressions using wigxjpf.

    USAGE:
        evaluator = SpinNetworkEvaluator(max_two_j=200)
        result = evaluator.evaluate(canonical_terms)
        evaluator.cleanup()
    """

    def __init__(self, max_two_j: int = 200, backend: str = 'auto',
                 n_workers: Optional[int] = None, verbose: bool = True):
        """
        Initialize the evaluator with wigxjpf tables.

        Parameters:
        -----------
        max_two_j : int
            Maximum value of 2*j that will be used. Default 200 (j up to 100).

            WHY 2*j?: Spin can be half-integer
            (e.g., j = 1/2, 3/2, 5/2...). We store as 2*j to use integers:
            - j = 1/2  → 2*j = 1
            - j = 1    → 2*j = 2
            - j = 3/2  → 2*j = 3
            etc.

        backend : str
            Computation backend: 'auto', 'serial', or 'multiprocessing'
            - 'auto'            : resolves to 'serial' (the default). See the
                                  comment in the body for the measurements
                                  behind that choice.
            - 'serial'          : single-threaded.
            - 'multiprocessing' : split the outermost summation across CPU
                                  cores. Only worth it for very large sums
                                  (break-even ~350,000 terms); the evaluator
                                  times a pilot slice and falls back to serial
                                  below that. REQUIRES the calling script to
                                  guard its entry point with
                                  `if __name__ == "__main__":`, because the
                                  'spawn' start method re-imports __main__ in
                                  every worker. Automatically disabled in
                                  notebooks and interactive sessions, where
                                  that re-import cannot work.

        n_workers : int, optional
            Number of parallel workers for multiprocessing backend.
            Defaults to cpu_count() - 1.

        verbose : bool
            Print backend/table messages. Worker processes pass False so a
            parallel run does not emit one banner per core.

        MEMORY USAGE:
            The tables scale as O(max_two_j^2), so be mindful for large j.
            max_two_j=200 uses ~few MB.
            max_two_j=1000 uses ~hundreds of MB.
        """
        self.max_two_j = max_two_j
        self.initialized = False
        self.verbose = verbose
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        def say(msg):
            if verbose:
                print(msg)

        # Backend selection.
        #
        # 'auto' resolves to SERIAL, deliberately. Two measured reasons:
        #
        #   1. Parallelism has a ~1 s fixed cost here (spawning workers and
        #      re-allocating the wigxjpf tables in each). At ~2.6 us per
        #      summation term that is a break-even of roughly 350,000 terms --
        #      more than almost any real formula reaches. Below that, the
        #      "parallel" backend is slower, by up to three orders of
        #      magnitude. Run scripts/benchmark_backends.py to reproduce.
        #
        #   2. On macOS and Windows the 'spawn' start method re-imports the
        #      caller's __main__ inside every worker. A user script without an
        #      `if __name__ == "__main__":` guard therefore RE-RUNS ITSELF once
        #      per worker. Defaulting to that would be a trap.
        #
        # Pass backend='multiprocessing' explicitly for the genuinely large
        # multi-variable summations where it pays off (measured 2.5x at 2.7M
        # terms). Even then the evaluator times a pilot slice first and stays
        # serial unless the work clearly exceeds the startup cost.
        if backend == 'auto':
            self.backend = 'serial'
            say("Using serial backend "
                "(pass backend='multiprocessing' for very large summations)")
        elif backend == 'serial':
            self.backend = 'serial'
            say("Using serial backend (single-threaded)")
        else:
            self.backend = backend
            say(f"🚀 Using {backend} backend")

        say(f"Initializing wigxjpf tables for max 2j = {max_two_j}...")

        # Initialize for 6j symbols (wigner_type=6)
        wig.wig_table_init(max_two_j, 6)

        # Allocate temporary calculation arrays
        # These are reused across calculations for efficiency
        wig.wig_temp_init(max_two_j)

        self.initialized = True
        say("✓ Wigxjpf initialized and ready")

    def cleanup(self):
        """
        Free wigxjpf memory. Call this when done with all calculations.

        IMPORTANT: Always call this to avoid memory leaks!
        """
        if self.initialized:
            wig.wig_temp_free()
            wig.wig_table_free()
            self.initialized = False
            if getattr(self, "verbose", True):
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
        Compute θ(j1,j2,j3)^power using the full definition with factorials.

        Parameters:
        -----------
        j1, j2, j3 : numeric values (spin)
        power : float, exponent on the theta symbol

        Returns:
        --------
        tuple (sign_exponent, magnitude) where result = (-1)^sign_exponent * magnitude

        FORMULA:
            θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
            θ(j,k,l)^p = (-1)^{p(j+k+l)} × |magnitude|^p

        NUMERICAL STABILITY:
            For large spins (j > 100), uses log-gamma to avoid overflow.
        """
        # Work with actual j values (not 2*j)
        j = j1 if isinstance(j1, (int, float)) else float(j1)
        k = j2 if isinstance(j2, (int, float)) else float(j2)
        l = j3 if isinstance(j3, (int, float)) else float(j3)

        # Check triangular inequality - if violated, theta = 0
        if not (abs(j - k) <= l <= j + k):
            return (0.0, 0.0)  # (sign_exponent, magnitude) - magnitude 0 means result is 0

        # Sign exponent: (j+k+l) * power
        sign_exponent = (j + k + l) * power

        # For large spins, use log-gamma to avoid overflow
        # factorial(~170) starts overflowing Python floats
        max_spin = max(j, k, l)
        if max_spin > 50:
            from scipy.special import gammaln

            # Compute in log space to avoid overflow
            log_num = gammaln(j + k + l + 2)
            log_denom = (gammaln(j + k - l + 1) +
                        gammaln(j - k + l + 1) +
                        gammaln(-j + k + l + 1))

            # log(|θ|) = log_num - log_denom (magnitude only, no sign)
            log_theta = log_num - log_denom

            # Apply power to magnitude: |θ|^p = exp(p × log(|θ|))
            magnitude = math.exp(power * log_theta)
        else:
            # For small spins, use cached factorials (faster)
            numerator = cached_factorial(int(j + k + l + 1))
            denom1 = cached_factorial(int(j + k - l))
            denom2 = cached_factorial(int(j - k + l))
            denom3 = cached_factorial(int(-j + k + l))

            # |θ(j,k,l)| = numerator / (denom1 × denom2 × denom3)
            theta_magnitude = numerator / (denom1 * denom2 * denom3)

            # Apply power to magnitude
            magnitude = math.pow(theta_magnitude, power)

        # Return (sign_exponent, magnitude) tuple for combining with other terms
        return (sign_exponent, magnitude)

    def delta_symbol(self, j, power=1.0):
        """
        Compute Δ_j^power = [(-1)^(2j) × (2j+1)]^power

        Parameters:
        -----------
        j : numeric value or variable
        power : float, exponent

        Returns:
        --------
        tuple (sign_exponent, magnitude) where result = (-1)^sign_exponent * magnitude

        FORMULA:
            Δ_j = (-1)^(2j) × (2j+1)
            Δ_j^p = (-1)^{2jp} × (2j+1)^p

        NUMERICAL STABILITY:
            Simple formula, no overflow issues even for large j.
        """
        # Work with actual j value (not 2*j)
        j_val = j if isinstance(j, (int, float)) else float(j)

        dimension = 2 * j_val + 1
        # Sign exponent: 2j * power
        sign_exponent = 2 * j_val * power

        # Apply power to magnitude only
        magnitude = math.pow(dimension, power)

        return (sign_exponent, magnitude)

    def theta_symbol_vectorized(self, j1_arr, j2_arr, j3_arr, power=1.0):
        """
        Vectorized theta computation for arrays of j values using full factorial formula.

        Parameters:
        -----------
        j1_arr, j2_arr, j3_arr : array-like of numeric values
        power : float

        Returns:
        --------
        np.ndarray : Array of theta values

        FORMULA:
            θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
        """
        j1_arr = np.asarray(j1_arr, dtype=float)
        j2_arr = np.asarray(j2_arr, dtype=float)
        j3_arr = np.asarray(j3_arr, dtype=float)

        # Check triangular inequality
        valid = (np.abs(j1_arr - j2_arr) <= j3_arr) & (j3_arr <= j1_arr + j2_arr)

        # Initialize result array
        result = np.zeros_like(j1_arr, dtype=float)

        # Only compute for valid triangles
        if np.any(valid):
            j_valid = j1_arr[valid]
            k_valid = j2_arr[valid]
            l_valid = j3_arr[valid]

            # Calculate sign: (-1)^(j+k+l)
            signs = np.power(-1.0, j_valid + k_valid + l_valid)

            # Calculate factorial terms using log-gamma for numerical stability
            # log(n!) = log(Γ(n+1)) → n! = exp(log(Γ(n+1)))
            # For large arrays, this is MUCH faster than computing factorials individually
            from scipy.special import gammaln

            # log(numerator) = log((j+k+l+1)!)
            log_num = gammaln(j_valid + k_valid + l_valid + 2)

            # log(denominator) = log((j+k-l)!) + log((j-k+l)!) + log((-j+k+l)!)
            log_denom = (gammaln(j_valid + k_valid - l_valid + 1) +
                        gammaln(j_valid - k_valid + l_valid + 1) +
                        gammaln(-j_valid + k_valid + l_valid + 1))

            # θ(j,k,l) = sign × exp(log_num - log_denom)
            theta_values = signs * np.exp(log_num - log_denom)

            # Apply power
            result[valid] = np.power(theta_values, power)

        return result

    def delta_symbol_vectorized(self, j_arr, power=1.0):
        """
        Vectorized delta computation for arrays of j values.

        Parameters:
        -----------
        j_arr : array-like of numeric values
        power : float

        Returns:
        --------
        np.ndarray : Array of delta values

        FORMULA:
            Δ_j = (-1)^(2j) × (2j+1)
            Δ_j^p = [(-1)^(2j) × (2j+1)]^p

        NUMERICAL STABILITY:
            Simple formula, no overflow issues.
        """
        j_arr = np.asarray(j_arr, dtype=float)
        dimensions = 2 * j_arr + 1
        signs = np.power(-1.0, 2 * j_arr)

        # Δ_j = (-1)^(2j) × (2j+1)
        delta_values = signs * dimensions

        # Apply power
        return np.power(delta_values, power)

    def wigner_6j(self, j1, j2, j3, j4, j5, j6, power=1.0):
        """
        Compute Wigner 6j symbol using wigxjpf.

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
        # Track sign exponents and magnitudes separately to avoid complex numbers
        pre_sign_exponent = 0.0
        pre_magnitude = 1.0

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
                result = self._evaluate_coefficient(c, {})
                # Handle (sign_exponent, magnitude) tuple
                if isinstance(result, tuple):
                    sign_exp, mag = result
                    pre_sign_exponent += sign_exp
                    pre_magnitude *= mag
                else:
                    pre_magnitude *= result

        # Step 3: If no summations, we're done
        if not sum_vars:
            # Combine sign and magnitude
            total_sign = (-1.0) ** int(round(pre_sign_exponent))
            return total_sign * pre_magnitude

        # Step 4: Perform summation (supports N variables)
        print(f"  Computing summation over {len(sum_vars)} variable(s)...")

        # Choose evaluation method based on backend and summation size
        if self.backend == 'serial':
            sum_result = self._evaluate_sum_serial(coeffs, sum_vars)
        elif self.backend == 'multiprocessing':
            sum_result = self._evaluate_sum_parallel(coeffs, sum_vars)
        else:
            # Fallback to serial
            sum_result = self._evaluate_sum_serial(coeffs, sum_vars)

        # Step 5: Combine pre-factor and sum
        # Pre-factor sign is computed here, sum already has sign resolved inside
        pre_sign = (-1.0) ** int(round(pre_sign_exponent))
        total = pre_sign * pre_magnitude * sum_result
        return total

    def _evaluate_sum_serial(self, coeffs, sum_vars):
        """
        Serial evaluation of N-variable summation.

        Uses itertools.product to generate all combinations of sum indices.
        """
        # Create list of (var_name, range) tuples
        var_names = list(sum_vars.keys())
        ranges = [range(min_val, max_val + 1) for min_val, max_val in sum_vars.values()]

        # Calculate total iterations for progress
        total_iters = 1
        for min_val, max_val in sum_vars.values():
            total_iters *= (max_val - min_val + 1)

        print(f"    Total iterations: {total_iters:,}")

        sum_result = 0.0
        count = 0

        # Generate all combinations using itertools.product
        for sum_values in itertools.product(*ranges):
            # Create substitution dict {var_name: value}
            substitutions = dict(zip(var_names, sum_values))

            # Evaluate all coefficients that depend on sum variables
            # Track sign exponents and magnitudes separately
            term_sign_exponent = 0.0
            term_magnitude = 1.0

            for c in coeffs:
                if not isinstance(c, dict):
                    continue

                typ = c.get("type")
                if typ == "sum":
                    continue

                depends_on_sum = self._depends_on_sum_var(c, sum_vars.keys())

                if depends_on_sum:
                    result = self._evaluate_coefficient(c, substitutions)
                    # Handle (sign_exponent, magnitude) tuple
                    if isinstance(result, tuple):
                        sign_exp, mag = result
                        term_sign_exponent += sign_exp
                        term_magnitude *= mag
                    else:
                        term_magnitude *= result

            # Combine sign and magnitude for this term
            # The total sign exponent should be an integer (spin network rule)
            term_sign = (-1.0) ** int(round(term_sign_exponent))
            term_value = term_sign * term_magnitude

            sum_result += term_value
            count += 1

            # Progress reporting for large sums
            if total_iters > 1000 and count % max(1, total_iters // 10) == 0:
                print(f"    Progress: {count:,}/{total_iters:,} ({100*count/total_iters:.1f}%)")

        return sum_result

    def _evaluate_sum_parallel(self, coeffs, sum_vars):
        """
        Parallel evaluation of N-variable summation using multiprocessing.

        Chunks the summation space and evaluates chunks in parallel.
        """
        # Create list of (var_name, range) tuples
        var_names = list(sum_vars.keys())
        ranges = [range(min_val, max_val + 1) for min_val, max_val in sum_vars.values()]

        # Generate all combinations
        all_combinations = list(itertools.product(*ranges))
        total_iters = len(all_combinations)

        print(f"    Total iterations: {total_iters:,}")
        print(f"    Using {self.n_workers} parallel workers")

        # Chunk the combinations for parallel processing
        chunk_size = max(1, total_iters // (self.n_workers * 4))  # 4x workers for load balancing
        chunks = [all_combinations[i:i + chunk_size] for i in range(0, total_iters, chunk_size)]

        print(f"    Split into {len(chunks)} chunks of ~{chunk_size} iterations each")

        # The chunk worker MUST be a module-level function, not a closure.
        # This code previously defined evaluate_chunk() inline and handed it to
        # Pool.map; nested functions cannot be pickled, so it raised as soon as
        # it was reached. It went unnoticed because 'auto' used to select the
        # JAX backend, which silently fell back to the serial path (JAX has
        # since been removed entirely: it cannot trace the wigxjpf C library).
        if not _multiprocessing_is_usable():
            print("    Multiprocessing unavailable in this context; using serial.")
            return self._evaluate_sum_serial(coeffs, sum_vars)

        tasks = [
            (coeffs, sum_vars, var_names, chunk, self.max_two_j)
            for chunk in chunks
        ]
        try:
            with Pool(
                processes=min(self.n_workers, len(chunks)),
                initializer=_parallel_worker_init,
                initargs=(self.max_two_j,),
            ) as pool:
                chunk_results = pool.map(_legacy_worker_evaluate_chunk, tasks)
        except Exception as exc:
            # Never fail the calculation because parallelism is unavailable.
            print(f"    Parallel evaluation unavailable ({exc}); using serial.")
            return self._evaluate_sum_serial(coeffs, sum_vars)

        sum_result = sum(chunk_results)
        return sum_result

    def _depends_on_sum_var(self, coeff: Dict, sum_var_names: set) -> bool:
        """Check if a coefficient depends on any summation variable."""
        # Check in args
        args = coeff.get("args", ())
        for arg in args:
            if isinstance(arg, str) and arg in sum_var_names:
                return True

        # Check in fixed dict (including nested structures for sign coefficients)
        fixed = coeff.get("fixed", {})
        for val in fixed.values():
            if isinstance(val, str) and val in sum_var_names:
                return True
            elif isinstance(val, (list, tuple)):
                # For sign coefficients: {"args": [('-', 'F_1'), ('+', 'F_2'), ...]}
                for item in val:
                    if isinstance(item, (list, tuple)):
                        # Each item is (sign, value) tuple
                        for sub_item in item:
                            if isinstance(sub_item, str) and sub_item in sum_var_names:
                                return True
                    elif isinstance(item, str) and item in sum_var_names:
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
            # Return (0, value) tuple - no sign exponent
            return (0.0, float(coeff.get("value", 1)))

        elif typ == "theta":
            args = coeff.get("args", ())
            power = coeff.get("power", 1)

            # Substitute variables
            j1, j2, j3 = args
            j1 = substitutions.get(j1, j1) if isinstance(j1, str) else j1
            j2 = substitutions.get(j2, j2) if isinstance(j2, str) else j2
            j3 = substitutions.get(j3, j3) if isinstance(j3, str) else j3

            # Returns (sign_exponent, magnitude) tuple
            return self.theta_symbol(j1, j2, j3, power)

        elif typ == "delta":
            fixed = coeff.get("fixed", {})
            j = fixed.get("j")
            power = coeff.get("power", 1)

            # Substitute variable
            j = substitutions.get(j, j) if isinstance(j, str) else j

            # Returns (sign_exponent, magnitude) tuple
            return self.delta_symbol(j, power)

        elif typ == "W6j":
            args = coeff.get("args", ())
            power = coeff.get("power", 1)

            # Substitute variables
            j_vals = []
            for j in args:
                j_val = substitutions.get(j, j) if isinstance(j, str) else j
                j_vals.append(j_val)

            # W6j has no sign factor, return (0, value) tuple
            return (0.0, self.wigner_6j(*j_vals, power=power))

        elif typ == "sign":
            # Handle (-1)^{sum of terms}
            # Format: {"type": "sign", "fixed": {"args": [(sign, value), ...]}}
            fixed = coeff.get("fixed", {})
            args = fixed.get("args", [])

            # Compute the exponent
            exponent = 0
            for sgn, val in args:
                # Substitute symbolic variables
                val_subst = substitutions.get(val, val) if isinstance(val, str) else val

                # Apply sign (sgn can be '+', '-', or None which means '+')
                if sgn == '-':
                    exponent -= val_subst
                else:  # '+' or None
                    exponent += val_subst

            # Return (sign_exponent, 1.0) tuple - magnitude is 1
            return (exponent, 1.0)

        else:
            print(f"Warning: Unknown coefficient type '{typ}'")
            return (0.0, 1.0)

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

        # Take absolute value since spin network norms are positive by definition
        # (sign conventions in the calculation can produce negative values)
        final_result = abs(total_result)

        # Round to nearest integer (spin network norms are integers)
        rounded_result = round(final_result)

        print("\n" + "=" * 70)
        print(f"FINAL RESULT: {rounded_result}")
        print(f"(Raw value before rounding: {final_result:.15e})")
        # Handle complex results (take absolute value)
        if isinstance(total_result, complex):
            print(f"(Note: Raw result was complex {total_result}, took absolute value)")
        elif total_result < 0:
            print(f"(Note: Raw result was {total_result:.15e}, took absolute value)")
        print("=" * 70)

        return rounded_result


# ============================================================================
# FORMULA EVALUATOR
# ============================================================================

# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------
#
# These MUST live at module level. The previous implementation passed a nested
# closure to Pool.map, which is unpicklable and therefore could never have run
# on macOS (spawn start method). Module-level functions pickle by qualified
# name, so they work under both fork and spawn.
#
# Each worker process builds its own FormulaEvaluator once, in the Pool
# initializer, because allocating the wigxjpf tables is the expensive part and
# must not be repeated per task.

_WORKER_EVALUATOR: Optional["FormulaEvaluator"] = None


def _multiprocessing_is_usable() -> bool:
    """
    Is it safe to start a worker Pool in this process?

    On macOS (and Windows) Python's default start method is 'spawn', which
    re-imports the parent's __main__ module inside every child. That works for
    a normal script, but NOT when __main__ cannot be re-imported:

      * an interactive interpreter or a heredoc  (__main__.__file__ is absent
        or the literal '<stdin>')
      * a Jupyter notebook

    In those contexts each child fails to import __main__ and the Pool retries
    forever -- the process hangs rather than raising, so a try/except around
    Pool() cannot save us. This is the "multiprocessing freezes in Jupyter"
    problem noted in PARALLEL_ACCELERATION.md.

    We therefore check up front and fall back to serial evaluation, which is
    always correct, just slower.
    """
    try:
        import multiprocessing as _mp
        if _mp.get_start_method(allow_none=False) == "fork":
            # fork copies the parent wholesale; no re-import, always safe here.
            return True
    except Exception:
        return False

    main_module = sys.modules.get("__main__")
    main_file = getattr(main_module, "__file__", None)
    if not main_file:
        return False  # notebook / interactive / stdin
    try:
        return os.path.isfile(main_file)
    except Exception:
        return False


def _parallel_worker_init(max_two_j: int) -> None:
    """Pool initializer: build this process's evaluator exactly once."""
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = FormulaEvaluator(
        max_two_j=max_two_j, backend="serial", verbose=False
    )


def _parallel_worker_evaluate_chunk(task) -> float:
    """Evaluate one sub-range of the outermost summation."""
    formula, variables, _max_two_j, chunk = task
    return _WORKER_EVALUATOR._evaluate_here(formula, variables, outer_chunk=chunk)


def _parallel_worker_evaluate_variables(task) -> float:
    """Evaluate the whole formula for one set of variable bindings."""
    formula, variables = task
    return _WORKER_EVALUATOR._evaluate_here(formula, variables)


def _legacy_worker_evaluate_chunk(task) -> float:
    """
    Evaluate one chunk of the Cartesian summation product for the legacy
    canonical-term path (SpinNetworkEvaluator.evaluate), used by scripts/.

    Replaces the unpicklable closure that made the multiprocessing backend
    unusable. All arguments are plain dicts/lists/tuples, so they pickle under
    both the fork and spawn start methods.
    """
    coeffs, sum_vars, var_names, chunk, _max_two_j = task
    ev = _WORKER_EVALUATOR._ev
    sum_var_names = set(sum_vars.keys())

    chunk_sum = 0.0
    for sum_values in chunk:
        substitutions = dict(zip(var_names, sum_values))

        # Sign exponents and magnitudes are tracked separately so that the
        # (-1)^... factors combine exactly instead of through repeated
        # floating-point multiplication.
        term_sign_exponent = 0.0
        term_magnitude = 1.0
        for c in coeffs:
            if not isinstance(c, dict) or c.get("type") == "sum":
                continue
            if not ev._depends_on_sum_var(c, sum_var_names):
                continue
            result = ev._evaluate_coefficient(c, substitutions)
            if isinstance(result, tuple):
                sign_exp, mag = result
                term_sign_exponent += sign_exp
                term_magnitude *= mag
            else:
                term_magnitude *= result

        chunk_sum += ((-1.0) ** int(round(term_sign_exponent))) * term_magnitude
    return chunk_sum


def _sanitize_primes(formula: str) -> str:
    """Replace prime notation in variable names with _p, skipping string literals.

    e.g. z' -> z_p, n'' -> n_p_p, but 'F_2' string literals are untouched.
    """
    result = []
    i = 0
    n = len(formula)
    while i < n:
        c = formula[i]
        if c in ('"', "'"):
            prev_is_word = bool(result) and (result[-1].isalnum() or result[-1] == '_')
            if prev_is_word:
                # Prime suffix on a variable name
                j = i
                while j < n and formula[j] == "'":
                    j += 1
                result.append("_p" * (j - i))
                i = j
            else:
                # String literal: copy until matching closing delimiter
                delim = c
                result.append(c)
                i += 1
                while i < n:
                    result.append(formula[i])
                    if formula[i] == delim:
                        i += 1
                        break
                    i += 1
        else:
            result.append(c)
            i += 1
    return "".join(result)


class FormulaEvaluator:
    """
    Evaluate arbitrary spin network formulas from a string expression.

    Supported functions in formula strings:
      theta(j1, j2, j3)              - Theta symbol (signed)
      delta(j)                       - Delta symbol (signed)
      deltatheta(c, s, t)             - Δ(c)/Θ(c,s,t), 0 if inadmissible (used
                                        by calculate_probability())
      safe_div(num, den)              - num/den, 0 if den == 0
      W6j(j1, j2, j3, j4, j5, j6)   - Wigner 6j symbol
      Sum('F', min, max, lambda F: …) - Summation over half/integer steps
      Standard math: sqrt, abs, exp, log, sin, cos, pi, ...

    EXAMPLE:
        fe = FormulaEvaluator()
        fe.evaluate("theta(1, 1, 1)")
        fe.evaluate("Sum('F', 0, 2, lambda F: delta(F) * W6j(1, 2, F, 1, 2, 3))")
        fe.evaluate("theta(j, j, 0) * W6j(j, j, 0, j, j, 0)", variables={"j": 1.0})
        fe.cleanup()
    """

    def __init__(self, max_two_j: int = 200, backend: str = 'auto',
                 verbose: bool = True):
        self._ev = SpinNetworkEvaluator(max_two_j, backend=backend,
                                        verbose=verbose)
        self._base_namespace = self._build_namespace()

    def evaluate_many(
        self,
        formula: str,
        variable_sets: List[Optional[Dict[str, float]]],
    ) -> List[float]:
        """
        Evaluate one formula for many variable bindings, in parallel.

        This is the embarrassingly-parallel case -- scanning a formula over a
        range of spin assignments -- and it is what Formula.evaluate_batch()
        uses. Unlike splitting a single summation, no linearity argument is
        needed here: the evaluations are completely independent.

        Returns
        -------
        list[float]
            Signed values, one per entry of variable_sets, in the same order.
        """
        formula = _sanitize_primes(formula)

        serial = (
            self._ev.backend == "serial"
            or len(variable_sets) < 2
            or self._ev.n_workers < 2
            or not _multiprocessing_is_usable()
        )
        if not serial:
            tasks = [(formula, v) for v in variable_sets]
            try:
                with Pool(
                    processes=min(self._ev.n_workers, len(tasks)),
                    initializer=_parallel_worker_init,
                    initargs=(self._ev.max_two_j,),
                ) as pool:
                    return [float(x) for x in
                            pool.map(_parallel_worker_evaluate_variables, tasks)]
            except Exception:
                pass  # fall through to the serial path below

        return [self._evaluate_here(formula, v) for v in variable_sets]

    def _build_namespace(self) -> dict:
        ev = self._ev

        def theta(j1, j2, j3, power=1.0):
            sign_exp, mag = ev.theta_symbol(j1, j2, j3, power)
            return ((-1.0) ** int(round(sign_exp))) * mag

        def delta(j, power=1.0):
            sign_exp, mag = ev.delta_symbol(j, power)
            return ((-1.0) ** int(round(sign_exp))) * mag

        def deltatheta(c, s, t, power=1.0):
            """Δ(c) / Θ(c, s, t)^power for a reconnection triplet.

            Returns exactly 0 when Θ is inadmissible (triangle inequality
            violated) instead of raising a division error — this is the
            physical convention used for transition probabilities: an
            inadmissible reconnection contributes zero.
            """
            th_sign, th_mag = ev.theta_symbol(c, s, t, power)
            if th_mag == 0.0:
                return 0.0
            d_sign, d_mag = ev.delta_symbol(c, power)
            sign_exp = d_sign - th_sign
            return ((-1.0) ** int(round(sign_exp))) * (d_mag / th_mag)

        def safe_div(numerator, denominator):
            """numerator / denominator, defined as 0 when denominator == 0.

            A norm of exactly zero means the associated state/transition is
            physically forbidden, so the probability is 0 rather than
            undefined.
            """
            return 0.0 if denominator == 0.0 else numerator / denominator

        def W6j(j1, j2, j3, j4, j5, j6, power=1.0):
            return ev.wigner_6j(j1, j2, j3, j4, j5, j6, power)

        def Sum(var_name, min_val, max_val, func):
            """Summation with integer step (matches spin network coupling rule)."""
            total = 0.0
            v = float(min_val)
            while v <= max_val + 1e-9:
                total += func(v)
                v += 1.0
            return total

        ns = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
        ns.update({
            'theta': theta,
            'delta': delta,
            'deltatheta': deltatheta,
            'safe_div': safe_div,
            'W6j': W6j,
            'Sum': Sum,
            'abs': abs,
            'round': round,
            'max': max,
            'min': min,
        })
        return ns

    def evaluate(self, formula: str, variables: Optional[Dict[str, float]] = None) -> float:
        """
        Evaluate a formula string numerically, preserving its sign.

        SIGN CONVENTION
        ---------------
        This returns the SIGNED value. Individual factors (theta, delta, the
        (-1)^... prefactors) legitimately carry signs and must keep them all
        the way through the computation, because they cancel against each
        other. Taking the modulus is a property of the *norm*, not of formula
        evaluation, so it is applied once at the very end by
        Formula.evaluate_numeric() / evaluate_batch() in src/api.py -- right
        before the norm is handed back to the caller.

        (This function used to return abs(...), which made every sign error in
        the reduction pipeline invisible and prevented callers from inspecting
        intermediate quantities.)

        Parameters
        ----------
        formula : str
            Expression using theta, delta, W6j, Sum, and math functions.
        variables : dict, optional
            Variable bindings, e.g. {"j": 1.0, "k": 0.5}

        Returns
        -------
        float
            The signed value of the expression.
        """
        formula = _sanitize_primes(formula)

        # 'serial' means "evaluate it right here"; anything else may be
        # distributed across processes if the formula's shape allows it.
        if self._ev.backend != "serial":
            parallel = self._try_parallel_evaluate(formula, variables)
            if parallel is not None:
                return parallel

        return self._evaluate_here(formula, variables)

    # ------------------------------------------------------------------
    # Serial core
    # ------------------------------------------------------------------

    def _evaluate_here(
        self,
        formula: str,
        variables: Optional[Dict[str, float]] = None,
        outer_chunk: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Evaluate the (already prime-sanitised) formula in this process.

        Parameters
        ----------
        outer_chunk : (lo, hi), optional
            Restrict the OUTERMOST Sum to this sub-range instead of its full
            range. Nested sums are unaffected. This is how a worker process
            evaluates its slice of a parallel run; see _try_parallel_evaluate
            for why summing the slices reproduces the whole.
        """
        ns = dict(self._base_namespace)
        if outer_chunk is not None:
            ns["Sum"] = self._make_chunked_sum(outer_chunk)
        if variables:
            # Sanitize the variable NAMES the same way the formula string is
            # sanitized below.  Graph edge labels may contain prime notation
            # (e.g. "n''"), but the formula string uses the Python-safe form
            # ("n_p_p"); without this, eval() cannot find the variable and
            # raises NameError.
            ns.update({_sanitize_primes(k): v for k, v in variables.items()})
        # Pass ns as globals (not locals) so that lambdas created inside eval
        # can resolve free variables (round, theta, W6j, ...) through their
        # __globals__, which is always the globals dict, never the locals dict.
        ns["__builtins__"] = {"__import__": None}
        try:
            return float(eval(formula, ns))
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {e}") from e

    @staticmethod
    def _make_chunked_sum(chunk):
        """
        Build a Sum() that restricts only its FIRST invocation to `chunk`.

        The first Sum call encountered during evaluation is the outermost one
        (a nested Sum can only be reached by calling the outer sum's lambda).
        We mark the outer call as consumed *before* invoking the lambda, so
        nested sums run over their full ranges as usual.
        """
        state = {"outer_consumed": False}
        lo_chunk, hi_chunk = chunk

        def Sum(var_name, min_val, max_val, func):
            if state["outer_consumed"]:
                lo, hi = float(min_val), float(max_val)
            else:
                state["outer_consumed"] = True
                lo, hi = float(lo_chunk), float(hi_chunk)
            total = 0.0
            v = lo
            while v <= hi + 1e-9:
                total += func(v)
                v += 1.0
            return total

        return Sum

    # ------------------------------------------------------------------
    # Parallel evaluation
    # ------------------------------------------------------------------

    # Fixed cost of going parallel: spawning the workers AND re-allocating the
    # wigxjpf tables inside each one. Measured at 0.97 s for 7 workers at
    # max_two_j=200 on an 8-core arm64 macOS box (see
    # scripts/benchmark_backends.py, which reproduces this number). It is
    # dominated by process startup, so it varies little with problem size.
    #
    # This is large: at ~2.6 us per summation term, roughly 435,000 terms of
    # serial work are needed before parallelism breaks even. A naive threshold
    # of a few dozen iterations -- which is what an untested implementation
    # tends to pick -- makes the "parallel" backend three orders of magnitude
    # SLOWER than serial on ordinary problems.
    _PARALLEL_STARTUP_SECONDS = 1.0

    # Safety margin on the break-even estimate: only parallelise when the
    # predicted saving clearly exceeds the startup cost.
    _PARALLEL_SPEEDUP_MARGIN = 1.5

    # How many outer-sum values to evaluate when timing the pilot.
    _PILOT_ITERATIONS = 2

    def _try_parallel_evaluate(
        self, formula: str, variables: Optional[Dict[str, float]]
    ) -> Optional[float]:
        """
        Attempt to evaluate `formula` by splitting its outermost summation
        across worker processes.

        WHY THIS IS CORRECT
        -------------------
        We only do this when the whole expression is a *product* in which the
        outermost Sum appears exactly once (checked with the ast module in
        _outer_sum_is_a_linear_factor). The expression is then

            value = C * S,      S = sum_{v in [lo, hi]} f(v)

        with C independent of v. Splitting [lo, hi] into disjoint chunks and
        summing the per-chunk evaluations gives

            sum_chunks C * S_chunk = C * sum_chunks S_chunk = C * S

        which is exactly the serial result. If the Sum is used non-linearly
        (raised to a power, in a denominator, passed as a function argument --
        as happens in the safe_div() form that calculate_probability()
        produces) the identity does not hold, so we return None and the caller
        falls back to serial rather than risk a wrong number.

        Returns
        -------
        float or None
            None means "not parallelisable, evaluate serially".
        """
        if not _multiprocessing_is_usable():
            return None

        if not self._outer_sum_is_a_linear_factor(formula):
            return None

        bounds = self._probe_outer_sum(formula, variables)
        if bounds is None:
            return None

        lo, hi = bounds
        n_iterations = int(math.floor(hi - lo + 1e-9)) + 1

        n_workers = max(1, min(self._ev.n_workers, n_iterations))
        if n_workers < 2:
            return None

        # Decide with a timed pilot rather than an iteration count.
        #
        # An iteration count is the wrong measure because the outer sum may be
        # short while each of its terms contains deeply nested sums -- the
        # expensive case this exists for. Timing a couple of real outer values
        # and extrapolating captures the nesting, and self-calibrates to the
        # machine instead of hard-coding one box's speed.
        pilot_n = min(self._PILOT_ITERATIONS, n_iterations)
        pilot_start = time.perf_counter()
        try:
            self._evaluate_here(
                formula, variables, outer_chunk=(lo, lo + pilot_n - 1)
            )
        except Exception:
            return None
        pilot_seconds = time.perf_counter() - pilot_start

        estimated_serial = pilot_seconds * (n_iterations / pilot_n)
        # Parallel wins when  startup + T/n < T, i.e. T > startup*n/(n-1).
        break_even = (
            self._PARALLEL_STARTUP_SECONDS
            * n_workers / (n_workers - 1)
            * self._PARALLEL_SPEEDUP_MARGIN
        )
        if estimated_serial < break_even:
            return None

        # Split [lo, hi] into n_workers contiguous chunks on the integer grid
        # of offsets from lo (the summation steps by 1).
        per_worker = math.ceil(n_iterations / n_workers)
        chunks = []
        start = 0
        while start < n_iterations:
            stop = min(start + per_worker, n_iterations)
            chunks.append((lo + start, lo + stop - 1))
            start = stop

        tasks = [
            (formula, variables, self._ev.max_two_j, chunk) for chunk in chunks
        ]
        try:
            with Pool(
                processes=len(chunks),
                initializer=_parallel_worker_init,
                initargs=(self._ev.max_two_j,),
            ) as pool:
                partials = pool.map(_parallel_worker_evaluate_chunk, tasks)
        except Exception:
            # Any multiprocessing problem (unavailable start method, pickling,
            # a sandbox that forbids fork) must degrade to a correct serial
            # answer rather than propagate.
            return None

        return float(sum(partials))

    @staticmethod
    def _outer_sum_is_a_linear_factor(formula: str) -> bool:
        """
        True if the formula's outermost Sum(...) occurs exactly once and only
        as a factor of a multiplication chain, so that chunking it is exact.

        See _try_parallel_evaluate for the algebra this guarantees.
        """
        try:
            tree = ast.parse(formula, mode="eval")
        except SyntaxError:
            return False

        # Record each Sum call together with its chain of ancestors.
        parents: Dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[child] = node

        def is_sum_call(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Sum"
            )

        def enclosing_sum_count(node):
            """How many Sum calls this node sits inside."""
            count = 0
            cur = parents.get(node)
            while cur is not None:
                if is_sum_call(cur):
                    count += 1
                cur = parents.get(cur)
            return count

        top_level_sums = [
            node
            for node in ast.walk(tree)
            if is_sum_call(node) and enclosing_sum_count(node) == 0
        ]
        if len(top_level_sums) != 1:
            # Zero sums: nothing to split. More than one: the "first call"
            # identified at run time would be ambiguous under reordering.
            return False

        # Every ancestor up to the root must be a multiplication, so the sum's
        # value enters the result linearly.
        node = top_level_sums[0]
        cur = parents.get(node)
        while cur is not None and not isinstance(cur, ast.Expression):
            if not (isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Mult)):
                return False
            cur = parents.get(cur)
        return True

    def _probe_outer_sum(
        self, formula: str, variables: Optional[Dict[str, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        Discover the outermost Sum's [lo, hi] without doing the work.

        We evaluate the formula with Sum replaced by a recorder that captures
        the bounds and returns 0.0 *without* calling the lambda -- so nested
        sums are never entered and the probe is cheap.
        """
        recorded: Dict[str, float] = {}

        def probe_sum(var_name, min_val, max_val, func):
            if not recorded:
                recorded["lo"] = float(min_val)
                recorded["hi"] = float(max_val)
            return 0.0

        ns = dict(self._base_namespace)
        ns["Sum"] = probe_sum
        if variables:
            ns.update({_sanitize_primes(k): v for k, v in variables.items()})
        ns["__builtins__"] = {"__import__": None}
        try:
            eval(formula, ns)
        except Exception:
            return None

        if not recorded:
            return None
        return recorded["lo"], recorded["hi"]

    def cleanup(self):
        self._ev.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def evaluate_spin_network(canonical_terms: List[Dict], max_two_j: int = 200,
                          backend: str = 'auto', n_workers: Optional[int] = None) -> float:
    """
    Convenience function to evaluate spin network in one call.

    Parameters:
    -----------
    canonical_terms : output from canonicalise_terms()
    max_two_j : maximum 2*j value expected
    backend : str
        Computation backend: 'auto', 'serial', or 'multiprocessing'
    n_workers : int, optional
        Number of parallel workers (for multiprocessing backend)

    Returns:
    --------
    float : numerical result

    EXAMPLE:
        # Auto-select best backend
        result = evaluate_spin_network(canon_terms, max_two_j=100)

        # Force multiprocessing with 8 workers
        result = evaluate_spin_network(canon_terms, max_two_j=100,
                                      backend='multiprocessing', n_workers=8)

        # Force CPU parallelism (large summations only)
        result = evaluate_spin_network(canon_terms, max_two_j=100, backend='multiprocessing')
    """
    evaluator = SpinNetworkEvaluator(max_two_j, backend=backend, n_workers=n_workers)
    try:
        result = evaluator.evaluate(canonical_terms)
        return result
    finally:
        evaluator.cleanup()


def benchmark_backends(canonical_terms: List[Dict], max_two_j: int = 200) -> Dict[str, float]:
    """
    Benchmark all available backends and compare performance.

    Parameters:
    -----------
    canonical_terms : output from canonicalise_terms()
    max_two_j : maximum 2*j value expected

    Returns:
    --------
    dict : {backend_name: execution_time_seconds}

    EXAMPLE:
        times = benchmark_backends(canon_terms)
        # Output: {'serial': 5.23, 'multiprocessing': 1.45}
    """
    import time

    results = {}
    backends_to_test = ['serial', 'multiprocessing']


    print("\n" + "=" * 70)
    print("BACKEND PERFORMANCE BENCHMARK")
    print("=" * 70)

    for backend in backends_to_test:
        print(f"\nTesting {backend} backend...")
        evaluator = SpinNetworkEvaluator(max_two_j, backend=backend)

        try:
            start_time = time.time()
            result = evaluator.evaluate(canonical_terms)
            elapsed = time.time() - start_time

            results[backend] = elapsed
            print(f"✓ {backend}: {elapsed:.3f} seconds (result: {result:.6e})")

        except Exception as e:
            print(f"✗ {backend} failed: {e}")
            results[backend] = float('inf')

        finally:
            evaluator.cleanup()

    # Find best backend
    best_backend = min(results, key=results.get)
    best_time = results[best_backend]

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    for backend, elapsed in sorted(results.items(), key=lambda x: x[1]):
        if elapsed == float('inf'):
            print(f"  {backend}: FAILED")
        else:
            speedup = results['serial'] / elapsed if backend != 'serial' else 1.0
            print(f"  {backend}: {elapsed:.3f}s (speedup: {speedup:.2f}x)")

    print(f"\n🏆 Best backend: {best_backend} ({best_time:.3f}s)")
    print("=" * 70)

    return results
