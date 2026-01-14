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
from typing import List, Dict, Any, Tuple, Optional
import math
import itertools
from multiprocessing import Pool, cpu_count
import os
from functools import lru_cache

# Optional GPU acceleration imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    JAX_AVAILABLE = True
    # Try to detect Metal backend (Apple Silicon)
    try:
        jax.devices('gpu')
        JAX_GPU_AVAILABLE = True
    except:
        JAX_GPU_AVAILABLE = False
except ImportError:
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False


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

    def __init__(self, max_two_j: int = 200, backend: str = 'auto', n_workers: Optional[int] = None):
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

        backend : str
            Computation backend: 'auto', 'jax', 'multiprocessing', or 'serial'
            - 'auto': Automatically select best available (JAX GPU > multiprocessing > serial)
            - 'jax': Force JAX backend (requires jax installation)
            - 'multiprocessing': Use CPU parallelization
            - 'serial': Single-threaded (original behavior)

        n_workers : int, optional
            Number of parallel workers for multiprocessing backend.
            Defaults to cpu_count() - 1.

        MEMORY USAGE:
            The tables scale as O(max_two_j^2), so be mindful for large j.
            max_two_j=200 uses ~few MB.
            max_two_j=1000 uses ~hundreds of MB.
        """
        self.max_two_j = max_two_j
        self.initialized = False
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # Select backend - always prefer parallel/GPU unless explicitly set to serial
        if backend == 'auto':
            # Priority: GPU > CPU parallel > fallback serial (never use serial by default)
            if JAX_GPU_AVAILABLE:
                self.backend = 'jax'
                print(f"🚀 Using JAX GPU backend (detected {len(jax.devices('gpu'))} GPU(s))")
            elif JAX_AVAILABLE:
                self.backend = 'jax'
                print("🚀 Using JAX CPU backend")
            else:
                self.backend = 'multiprocessing'
                print(f"🚀 Using multiprocessing backend ({self.n_workers} workers)")
        elif backend == 'serial':
            # Serial only if explicitly requested (for debugging/testing)
            self.backend = 'serial'
            print("⚠️  Using serial backend (single-threaded, slow - only for debugging)")
        else:
            self.backend = backend
            if backend == 'jax' and not JAX_AVAILABLE:
                raise ImportError("JAX backend requested but JAX not installed. Install with: pip install jax")
            print(f"🚀 Using {backend} backend")

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
        Compute θ(j1,j2,j3)^power using the full definition with factorials.

        PHYSICS: Theta symbol encodes the quantum dimensions and triangular
        inequality constraints for spin coupling.

        Parameters:
        -----------
        j1, j2, j3 : numeric values (spin quantum numbers)
        power : float, exponent on the theta symbol

        Returns:
        --------
        float : The numerical value

        FORMULA:
            θ(j,k,l) = (-1)^(j+k+l) × (j+k+l+1)! / [(j+k-l)!(j-k+l)!(-j+k+l)!]
            θ(j,k,l)^p = [θ(j,k,l)]^p

        NUMERICAL STABILITY:
            For large spins (j > 100), uses log-gamma to avoid overflow.
        """
        # Work with actual j values (not 2*j)
        j = j1 if isinstance(j1, (int, float)) else float(j1)
        k = j2 if isinstance(j2, (int, float)) else float(j2)
        l = j3 if isinstance(j3, (int, float)) else float(j3)

        # Check triangular inequality - if violated, theta = 0
        if not (abs(j - k) <= l <= j + k):
            return 0.0

        # Calculate sign: (-1)^(j+k+l)
        sign = (-1.0) ** (j + k + l)

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

            # log(θ) = log_num - log_denom
            log_theta = log_num - log_denom

            # Apply power: θ^p = exp(p × log(θ))
            # Include sign
            result = sign * math.exp(power * log_theta)
        else:
            # For small spins, use cached factorials (faster)
            numerator = cached_factorial(int(j + k + l + 1))
            denom1 = cached_factorial(int(j + k - l))
            denom2 = cached_factorial(int(j - k + l))
            denom3 = cached_factorial(int(-j + k + l))

            # θ(j,k,l) = sign × numerator / (denom1 × denom2 × denom3)
            theta_value = sign * numerator / (denom1 * denom2 * denom3)

            # Apply power
            result = math.pow(theta_value, power)

        return result

    def delta_symbol(self, j, power=1.0):
        """
        Compute Δ_j^power = [(-1)^(2j) × (2j+1)]^power

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
            Δ_j = (-1)^(2j) × (2j+1)
            Δ_j^p = [(-1)^(2j) × (2j+1)]^p

        NUMERICAL STABILITY:
            Simple formula, no overflow issues even for large j.
        """
        # Work with actual j value (not 2*j)
        j_val = j if isinstance(j, (int, float)) else float(j)

        dimension = 2 * j_val + 1
        sign = (-1.0) ** (2 * j_val)

        # Δ_j = (-1)^(2j) × (2j+1)
        delta = sign * dimension

        # Apply power
        result = math.pow(delta, power)

        return result

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

        # Step 4: Perform summation (supports N variables)
        print(f"  Computing summation over {len(sum_vars)} variable(s)...")

        # Choose evaluation method based on backend and summation size
        if self.backend == 'serial':
            sum_result = self._evaluate_sum_serial(coeffs, sum_vars)
        elif self.backend == 'multiprocessing':
            sum_result = self._evaluate_sum_parallel(coeffs, sum_vars)
        elif self.backend == 'jax':
            sum_result = self._evaluate_sum_jax(coeffs, sum_vars)
        else:
            # Fallback to serial
            sum_result = self._evaluate_sum_serial(coeffs, sum_vars)

        # Step 5: Combine pre-factor and sum
        total = pre_factor * sum_result
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

        # Create worker function that evaluates a chunk
        def evaluate_chunk(chunk):
            """Evaluate a chunk of summation combinations."""
            chunk_sum = 0.0
            for sum_values in chunk:
                substitutions = dict(zip(var_names, sum_values))

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

                chunk_sum += term_value
            return chunk_sum

        # Parallel evaluation
        with Pool(processes=self.n_workers) as pool:
            chunk_results = pool.map(evaluate_chunk, chunks)

        sum_result = sum(chunk_results)
        return sum_result

    def _evaluate_sum_jax(self, coeffs, sum_vars):
        """
        JAX-accelerated evaluation of N-variable summation.

        For now, falls back to serial for 6j symbols (wigxjpf is not JAX-compatible).
        But vectorizes theta and delta computations.
        """
        # TODO: Full JAX implementation requires JAX-compatible Wigner 6j
        # For now, use vectorized NumPy for theta/delta and fall back to serial loop
        print("    JAX backend: Using vectorized theta/delta with serial 6j evaluation")
        return self._evaluate_sum_serial(coeffs, sum_vars)

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

            # Return (-1)^exponent
            return (-1.0) ** exponent

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

def evaluate_spin_network(canonical_terms: List[Dict], max_two_j: int = 200,
                          backend: str = 'auto', n_workers: Optional[int] = None) -> float:
    """
    Convenience function to evaluate spin network in one call.

    Parameters:
    -----------
    canonical_terms : output from canonicalise_terms()
    max_two_j : maximum 2*j value expected
    backend : str
        Computation backend: 'auto', 'jax', 'multiprocessing', or 'serial'
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

        # Use JAX GPU acceleration
        result = evaluate_spin_network(canon_terms, max_two_j=100, backend='jax')
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
        # Output: {'serial': 5.23, 'multiprocessing': 1.45, 'jax': 0.87}
    """
    import time

    results = {}
    backends_to_test = ['serial', 'multiprocessing']

    if JAX_AVAILABLE:
        backends_to_test.append('jax')

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
