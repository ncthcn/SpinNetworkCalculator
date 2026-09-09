"""
Tests for src/norm_reducer.py -- the post-reduction simplification layer.

WHY THIS MATTERS
----------------
This module sits between the graph reduction and the numerical evaluation, and
it is where the most error-prone bookkeeping lives: merging spins that a
Kronecker delta has identified, expanding the graph-level 6j into its
theta/delta/W6j definition, and collapsing equivalent factors.

The dangerous failure mode is **over-merging**: if the canonical key treats two
factors as identical when they are not, they are silently combined into a
single power and the answer is wrong with no error. The key test here is
therefore not "does the key stay the same under a symmetry" but the converse --
"do two factors that share a key really have the same numerical value".

For a C++ reader: pytest collects any `test_*` function; there is no
registration step. A coefficient is a plain Python dict with a "type" field,
which is how this codebase represents an algebraic factor.
"""

import itertools
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_reducer import (
    build_6j_coeff,
    build_delta_coeff,
    build_kronecker_coeff,
    build_sum_coeff,
    build_theta_coeff,
)
from src.norm_reducer import (
    apply_kroneckers,
    canonicalise_sign,
    canonicalise_terms,
    expand_6j_symbolic,
    reconstruct_terms_from_canonical,
    sort_mixed_args,
    theta_key,
    wigner_sixj_key,
)
from src.spin_evaluator import SpinNetworkEvaluator


@pytest.fixture(scope="module")
def ev():
    """One shared evaluator: wigxjpf keeps process-global C tables."""
    evaluator = SpinNetworkEvaluator(max_two_j=40, backend="serial", verbose=False)
    yield evaluator
    evaluator.cleanup()


def term(*coeffs):
    """Wrap coefficients in the term dict shape the module expects."""
    return {"coeffs": list(coeffs)}


def types_of(t):
    return [c.get("type") for c in t["coeffs"]]


# ===========================================================================
# apply_kroneckers -- union-find over spin equalities
# ===========================================================================

class TestApplyKroneckers:

    def test_numeric_contradiction_kills_the_term(self):
        """Kronecker(1, 2) demands j=1 and j=2 at once: the term is zero."""
        assert apply_kroneckers(term(build_kronecker_coeff(1, 2))) is None

    def test_consistent_numeric_equality_survives(self):
        """Kronecker(1, 1) is satisfiable and must not kill the term."""
        assert apply_kroneckers(term(build_kronecker_coeff(1, 1))) is not None

    def test_kronecker_coefficients_are_consumed(self):
        """Once applied, the constraints themselves must not remain as factors."""
        out = apply_kroneckers(term(
            build_kronecker_coeff("a", "b"),
            build_theta_coeff("a", "x", "y"),
        ))
        assert "Kronecker" not in types_of(out)

    def test_equal_labels_are_substituted_to_one_representative(self):
        """Kronecker(a, b) must rewrite every 'a' or 'b' to a single name."""
        out = apply_kroneckers(term(
            build_kronecker_coeff("a", "b"),
            build_theta_coeff("a", "x", "y"),
            build_theta_coeff("b", "x", "y"),
        ))
        thetas = [c for c in out["coeffs"] if c["type"] == "theta"]
        assert len(thetas) == 2
        first = thetas[0]["fixed"]["a"]
        second = thetas[1]["fixed"]["a"]
        assert first == second, (
            f"'a' and 'b' were identified but map to {first!r} and {second!r}"
        )
        assert first in ("a", "b")

    def test_equality_is_transitive(self):
        """a=b and b=c must put a, b and c all in the same class."""
        out = apply_kroneckers(term(
            build_kronecker_coeff("a", "b"),
            build_kronecker_coeff("b", "c"),
            build_theta_coeff("a", "x", "y"),
            build_theta_coeff("c", "x", "y"),
        ))
        thetas = [c for c in out["coeffs"] if c["type"] == "theta"]
        assert thetas[0]["fixed"]["a"] == thetas[1]["fixed"]["a"], (
            "a=b=c was not resolved to a single representative"
        )

    def test_duplicate_sums_are_merged_with_intersected_ranges(self):
        """
        Two summations over the same index must collapse into one whose range
        is the intersection -- a spin must satisfy both constraints.
        """
        out = apply_kroneckers(term(
            build_sum_coeff("F_1", {"Fmin": 0, "Fmax": 10, "parity": 0}),
            build_sum_coeff("F_1", {"Fmin": 4, "Fmax": 20, "parity": 0}),
        ))
        sums = [c for c in out["coeffs"] if c["type"] == "sum"]
        assert len(sums) == 1, f"expected the sums to merge, got {len(sums)}"
        assert sums[0]["range2"]["Fmin"] == 4
        assert sums[0]["range2"]["Fmax"] == 10

    def test_disjoint_sum_ranges_kill_the_term(self):
        """An empty summation range means the term contributes nothing."""
        out = apply_kroneckers(term(
            build_sum_coeff("F_1", {"Fmin": 0, "Fmax": 3, "parity": 0}),
            build_sum_coeff("F_1", {"Fmin": 8, "Fmax": 12, "parity": 0}),
        ))
        assert out is None

    def test_distinct_sum_indices_are_left_alone(self):
        out = apply_kroneckers(term(
            build_sum_coeff("F_1", {"Fmin": 0, "Fmax": 4, "parity": 0}),
            build_sum_coeff("F_2", {"Fmin": 0, "Fmax": 6, "parity": 0}),
        ))
        sums = [c for c in out["coeffs"] if c["type"] == "sum"]
        assert len(sums) == 2

    def test_is_idempotent(self):
        """
        Re-applying the pass must not change anything: the representatives are
        already fixed points of the substitution.
        """
        once = apply_kroneckers(term(
            build_kronecker_coeff("a", "b"),
            build_theta_coeff("a", "x", "y"),
        ))
        twice = apply_kroneckers({"coeffs": [dict(c) for c in once["coeffs"]]})
        assert types_of(once) == types_of(twice)
        assert [c.get("fixed") for c in once["coeffs"]] == \
               [c.get("fixed") for c in twice["coeffs"]]

    def test_unconstrained_term_passes_through_unchanged(self):
        original = term(build_theta_coeff(1, 1, 2), build_delta_coeff(1))
        out = apply_kroneckers(original)
        assert types_of(out) == ["theta", "delta"]


# ===========================================================================
# expand_6j_symbolic -- graph 6j -> theta/delta/W6j
# ===========================================================================

class TestExpand6jSymbolic:
    """
    The documented identity is

        {a b f; c d e}  =  (-1)^(-a-b-c-d+2f) * Delta_f
                           * Theta(a,b,e)^(1/2)  * Theta(c,d,e)^(1/2)
                           * Theta(b,c,f)^(-1/2) * Theta(a,d,f)^(-1/2)
                           * W6j(a,b,e,c,d,f)
    """

    def test_produces_the_seven_documented_factors(self):
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "f", "c", "d", "e"))
        assert [c["type"] for c in out] == [
            "sign", "delta", "theta", "theta", "theta", "theta", "W6j"
        ]

    def test_theta_powers_are_two_halves_and_two_negative_halves(self):
        """
        The four theta factors must come in a +1/2, +1/2, -1/2, -1/2 pattern.
        Getting a sign wrong here would rescale every norm by a theta ratio.
        """
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "f", "c", "d", "e"))
        powers = sorted(c["power"] for c in out if c["type"] == "theta")
        assert powers == [-0.5, -0.5, 0.5, 0.5]

    def test_theta_arguments_match_the_documented_identity(self):
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "f", "c", "d", "e"))
        thetas = {c["args"]: c["power"] for c in out if c["type"] == "theta"}
        assert thetas[("a", "b", "e")] == 0.5
        assert thetas[("c", "d", "e")] == 0.5
        assert thetas[("b", "c", "f")] == -0.5
        assert thetas[("a", "d", "f")] == -0.5

    def test_sign_exponent_matches_the_documented_identity(self):
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "f", "c", "d", "e"))
        sign = next(c for c in out if c["type"] == "sign")
        assert sign["fixed"]["args"] == [
            ("-", "a"), ("-", "b"), ("-", "c"), ("-", "d"), ("2", "f")
        ]

    def test_delta_is_on_the_summation_index_f(self):
        """Delta must carry f -- the recoupling index -- not one of the legs."""
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "F_1", "c", "d", "e"))
        delta = next(c for c in out if c["type"] == "delta")
        assert delta["fixed"]["j"] == "F_1"

    def test_w6j_argument_order_is_a_b_e_c_d_f(self):
        """
        The graph-level {a b f; c d e} becomes W6j(a, b, e, c, d, f): the
        summation index f moves to the last slot. A permutation bug here would
        silently evaluate a different (still valid) 6j symbol.
        """
        out = expand_6j_symbolic(build_6j_coeff("a", "b", "f", "c", "d", "e"))
        w6j = next(c for c in out if c["type"] == "W6j")["fixed"]
        assert (w6j["a"], w6j["b"], w6j["e"], w6j["c"], w6j["d"], w6j["f"]) == \
               ("a", "b", "e", "c", "d", "f")

    def test_accepts_uppercase_doubled_keys_as_a_fallback(self):
        """
        Coefficient builders store both 'a' and the doubled 'A'. The expansion
        must fall back to the uppercase key when the lowercase one is absent.
        """
        out = expand_6j_symbolic({
            "type": "6j",
            "fixed": {"A": 1, "B": 1, "F": 2, "C": 1, "D": 1, "E": 1},
            "power": 1,
        })
        w6j = next(c for c in out if c["type"] == "W6j")["fixed"]
        assert None not in w6j.values(), f"uppercase fallback failed: {w6j}"

    def test_expansion_evaluates_to_a_finite_number(self, ev):
        """End-to-end sanity: the expanded factors must be numerically usable."""
        out = expand_6j_symbolic(build_6j_coeff(1, 1, 1, 1, 1, 1))
        value = 1.0
        sign_exp = 0.0
        for c in out:
            if c["type"] == "theta":
                s, m = ev.theta_symbol(*c["args"], power=c["power"])
                sign_exp += s
                value *= m
            elif c["type"] == "delta":
                s, m = ev.delta_symbol(c["fixed"]["j"], power=c["power"])
                sign_exp += s
                value *= m
            elif c["type"] == "W6j":
                f = c["fixed"]
                value *= ev.wigner_6j(f["a"], f["b"], f["e"], f["c"], f["d"], f["f"])
        result = ((-1.0) ** int(round(sign_exp))) * value
        assert result == result, "expansion produced NaN"           # NaN != NaN
        assert abs(result) != float("inf")
        assert result != 0.0


# ===========================================================================
# canonicalise_sign -- collapsing (-1)^(...) exponents
# ===========================================================================

class TestCanonicaliseSign:

    def make(self, args):
        return canonicalise_sign({"type": "sign", "fixed": {"args": args}})

    def test_even_numeric_exponent_becomes_the_identity(self):
        """(-1)^4 = 1, so the factor should disappear entirely (None)."""
        assert self.make([("+", 4)]) is None

    def test_odd_numeric_exponent_becomes_minus_one(self):
        out = self.make([("+", 3)])
        assert out == {"type": "sign_value", "value": -1}

    def test_zero_exponent_is_the_identity(self):
        assert self.make([]) is None
        assert self.make([("+", 0)]) is None

    def test_minus_sign_parses_as_coefficient_minus_one(self):
        """(-1)^(-3) is odd, so this must be -1, not +1."""
        assert self.make([("-", 3)]) == {"type": "sign_value", "value": -1}

    def test_numeric_string_coefficient_is_honoured(self):
        """('2', 3) means 2*3 = 6, which is even."""
        assert self.make([("2", 3)]) is None

    def test_repeated_variable_is_merged_with_summed_coefficient(self):
        """(+a) + (+a) must become 2a, not two separate entries."""
        out = self.make([("+", "a"), ("+", "a")])
        assert out["fixed"]["args"] == [("+2", "a")]

    def test_variable_that_cancels_leaves_only_the_numeric_part(self):
        """(+a) + (-a) = 0, so only the numeric parity decides."""
        assert self.make([("+", "a"), ("-", "a"), ("+", 2)]) is None
        assert self.make([("+", "a"), ("-", "a"), ("+", 1)]) == \
               {"type": "sign_value", "value": -1}

    def test_symbolic_exponent_is_kept_symbolic(self):
        """With a surviving variable the sign cannot be resolved to +-1 yet."""
        out = self.make([("+", "a")])
        assert out["type"] == "sign"
        assert out["fixed"]["args"] == [("+", "a")]

    def test_variables_are_emitted_in_sorted_order(self):
        """Deterministic output: two runs must not differ by dict ordering."""
        out = self.make([("+", "z"), ("+", "a"), ("+", "m")])
        names = [v for _, v in out["fixed"]["args"]]
        assert names == sorted(names)


# ===========================================================================
# Canonical keys -- the over-merging hazard
# ===========================================================================

class TestThetaKey:

    def test_is_invariant_under_argument_permutation(self):
        """Theta is totally symmetric, so all 6 orderings share one key."""
        keys = {
            theta_key(build_theta_coeff(*p))
            for p in itertools.permutations([1, 2, 3])
        }
        assert len(keys) == 1

    def test_distinguishes_genuinely_different_arguments(self):
        assert theta_key(build_theta_coeff(1, 2, 3)) != \
               theta_key(build_theta_coeff(1, 2, 4))

    def test_distinguishes_different_powers(self):
        assert theta_key(build_theta_coeff(1, 2, 3, power=1)) != \
               theta_key(build_theta_coeff(1, 2, 3, power=2))

    def test_numbers_sort_before_symbols(self):
        _, args, _ = theta_key({"type": "theta", "args": ("z", 1, "a"), "power": 1})
        assert args == (1, "a", "z")

    def test_accepts_both_dict_and_tuple_forms(self):
        from_dict = theta_key({"type": "theta", "args": (1, 2, 3), "power": 1})
        from_tuple = theta_key(("theta", (1, 2, 3), 1))
        assert from_dict == from_tuple

    def test_rejects_unsupported_input(self):
        with pytest.raises(TypeError):
            theta_key(["theta", (1, 2, 3), 1])


class TestWignerSixjKey:
    """
    `wigner_sixj_key` merges 6j coefficients using the 24-element tetrahedral
    symmetry group (column permutations and upper/lower swaps in pairs of
    columns).

    The critical property is soundness, not completeness: if the key merges
    two symbols whose values differ, the canonicalisation silently produces a
    wrong power and the norm is wrong. The exhaustive test below checks
    exactly that, against the actual numerical 6j.
    """

    def w6j_coeff(self, args, power=1):
        a, b, e, c, d, f = args
        return {
            "type": "W6j",
            "fixed": {"a": a, "b": b, "e": e, "c": c, "d": d, "f": f},
            "power": power,
        }

    def test_column_permutation_gives_the_same_key(self):
        """Swapping two whole columns leaves the 6j symbol unchanged."""
        base = self.w6j_coeff((1, 2, 3, 4, 5, 6))
        # {j1 j2 j3; j4 j5 j6} -> {j2 j1 j3; j5 j4 j6}
        swapped = self.w6j_coeff((2, 1, 3, 5, 4, 6))
        assert wigner_sixj_key(base) == wigner_sixj_key(swapped)

    def test_upper_lower_swap_in_two_columns_gives_the_same_key(self):
        """{j1 j2 j3; j4 j5 j6} -> {j4 j5 j3; j1 j2 j6}"""
        base = self.w6j_coeff((1, 2, 3, 4, 5, 6))
        swapped = self.w6j_coeff((4, 5, 3, 1, 2, 6))
        assert wigner_sixj_key(base) == wigner_sixj_key(swapped)

    def test_key_is_stable_under_repeated_application(self):
        base = self.w6j_coeff((1, 2, 3, 4, 5, 6))
        _, canon, _ = wigner_sixj_key(base)
        assert wigner_sixj_key(self.w6j_coeff(canon)) == wigner_sixj_key(base)

    def test_distinguishes_different_powers(self):
        assert wigner_sixj_key(self.w6j_coeff((1, 2, 3, 4, 5, 6), power=1)) != \
               wigner_sixj_key(self.w6j_coeff((1, 2, 3, 4, 5, 6), power=2))

    def test_merged_symbols_always_have_equal_numerical_value(self, ev):
        """
        THE IMPORTANT ONE.

        Enumerate every 6-tuple of small spins, group them by canonical key,
        and verify that everything sharing a key really has the same 6j value.

        A spurious entry in the symmetry list would merge two distinct symbols
        here and this test would fail. Note the test does NOT assume the group
        is complete -- missing a symmetry only costs simplification, whereas an
        incorrect one corrupts the result.
        """
        spins = [0.0, 0.5, 1.0, 1.5]
        by_key = {}
        for args in itertools.product(spins, repeat=6):
            key = wigner_sixj_key(self.w6j_coeff(args))
            # wigner_sixj_key builds its tuple as (a, b, e, c, d, f); the
            # evaluator takes the six slots in that same positional order.
            value = ev.wigner_6j(*args)
            by_key.setdefault(key, []).append((args, value))

        offenders = []
        for key, entries in by_key.items():
            values = [v for _, v in entries]
            if max(values) - min(values) > 1e-12:
                offenders.append((key, entries[:4]))

        assert not offenders, (
            f"{len(offenders)} canonical key(s) merge 6j symbols with different "
            f"values -- the symmetry list is unsound. First: {offenders[0]}"
        )

    def test_actually_merges_something(self, ev):
        """
        Guard against the test above passing vacuously: if the key were simply
        the identity, nothing would ever merge and soundness would be trivial.
        """
        spins = [0.0, 0.5, 1.0, 1.5]
        keys = {wigner_sixj_key(self.w6j_coeff(a))
                for a in itertools.product(spins, repeat=6)}
        total = len(spins) ** 6
        assert len(keys) < total, "canonicalisation merged nothing at all"


class TestSortMixedArgs:

    def test_numbers_first_then_symbols_alphabetically(self):
        assert sort_mixed_args(("b", 2, "a", 1)) == (1, 2, "a", "b")

    def test_is_idempotent(self):
        once = sort_mixed_args(("b", 2, "a", 1))
        assert sort_mixed_args(once) == once


# ===========================================================================
# canonicalise_terms -- power accumulation and ordering
# ===========================================================================

class TestCanonicaliseTerms:

    def test_identical_thetas_accumulate_their_powers(self):
        out = canonicalise_terms([term(
            build_theta_coeff(1, 2, 3, power=1),
            build_theta_coeff(1, 2, 3, power=1),
        )])
        thetas = [c for c in out[0]["coeffs"] if c["type"] == "theta"]
        assert len(thetas) == 1
        assert thetas[0]["power"] == 2

    def test_permuted_thetas_are_recognised_as_the_same_factor(self):
        """theta(1,2,3) and theta(3,1,2) are the same symbol."""
        out = canonicalise_terms([term(
            build_theta_coeff(1, 2, 3),
            build_theta_coeff(3, 1, 2),
        )])
        thetas = [c for c in out[0]["coeffs"] if c["type"] == "theta"]
        assert len(thetas) == 1 and thetas[0]["power"] == 2

    def test_factors_that_cancel_are_dropped(self):
        """theta^(+1) * theta^(-1) = 1 and must not appear at all."""
        out = canonicalise_terms([term(
            build_theta_coeff(1, 2, 3, power=1),
            build_theta_coeff(1, 2, 3, power=-1),
        )])
        assert not [c for c in out[0]["coeffs"] if c["type"] == "theta"]

    def test_deltas_accumulate_by_spin(self):
        out = canonicalise_terms([term(
            build_delta_coeff(1.5, power=1),
            build_delta_coeff(1.5, power=2),
            build_delta_coeff(2.0, power=1),
        )])
        deltas = {c["fixed"]["j"]: c["power"]
                  for c in out[0]["coeffs"] if c["type"] == "delta"}
        assert deltas == {1.5: 3, 2.0: 1}

    def test_signs_are_merged_into_a_single_factor(self):
        out = canonicalise_terms([term(
            {"type": "sign", "fixed": {"args": [("+", 1)]}},
            {"type": "sign", "fixed": {"args": [("+", 1)]}},
        )])
        # (-1)^1 * (-1)^1 = (-1)^2 = 1, so nothing should survive.
        assert not [c for c in out[0]["coeffs"]
                    if c["type"] in ("sign", "sign_value")]

    def test_odd_merged_sign_becomes_an_explicit_minus_one(self):
        out = canonicalise_terms([term(
            {"type": "sign", "fixed": {"args": [("+", 1)]}},
            {"type": "sign", "fixed": {"args": [("+", 2)]}},
        )])
        signs = [c for c in out[0]["coeffs"] if c["type"] == "sign_value"]
        assert len(signs) == 1 and signs[0]["value"] == -1

    def test_constant_factors_are_ordered_before_the_summation(self):
        """
        Output shape must be  [constants] x Sum [F-dependent factors]  so the
        generated formula string nests correctly.
        """
        out = canonicalise_terms([term(
            build_sum_coeff("F_1", {"Fmin": 0, "Fmax": 4, "parity": 0}),
            build_theta_coeff(1, 2, 3),            # constant
            build_delta_coeff("F_1"),              # depends on the sum index
        )])
        kinds = types_of(out[0])
        assert "sum" in kinds
        sum_at = kinds.index("sum")
        theta_at = kinds.index("theta")
        delta_at = kinds.index("delta")
        assert theta_at < sum_at, "constant theta must precede the summation"
        assert delta_at > sum_at, "F-dependent delta must follow the summation"

    def test_factor_depending_on_the_sum_index_is_detected_through_args(self):
        out = canonicalise_terms([term(
            build_sum_coeff("F_1", {"Fmin": 0, "Fmax": 4, "parity": 0}),
            build_theta_coeff("F_1", 1, 1),
        )])
        kinds = types_of(out[0])
        assert kinds.index("theta") > kinds.index("sum")

    def test_empty_term_survives(self):
        out = canonicalise_terms([term()])
        assert out[0]["coeffs"] == []


class TestReconstructTerms:

    def test_round_trips_coefficients_unchanged(self):
        original = [term(build_theta_coeff(1, 2, 3), build_delta_coeff(1))]
        out = reconstruct_terms_from_canonical(original)
        assert len(out) == 1
        assert out[0]["coeffs"] == original[0]["coeffs"]

    def test_produces_a_fresh_list(self):
        """The rebuilt term must not alias the input list."""
        original = [term(build_theta_coeff(1, 2, 3))]
        out = reconstruct_terms_from_canonical(original)
        assert out[0]["coeffs"] is not original[0]["coeffs"]
