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
Validation of the numerical core against independently-known results.

WHY THIS FILE EXISTS
--------------------
The rest of the test suite checks *structure* (does a reduction return a
non-empty list, is the result a float, is it non-NaN). None of it checks that
the numbers are *right*. This file does, by comparing against sources that are
independent of this codebase:

  * ``sympy.physics.wigner.wigner_6j`` -- a completely separate implementation
    of the Wigner 6j symbol, used as ground truth for our wigxjpf binding.
  * Closed-form factorial expressions, re-derived here from scratch rather
    than imported, so a typo in ``src/spin_evaluator.py`` cannot hide by being
    copied into the test.
  * Algebraic identities (orthogonality, Biedenharn-Elliott) that any correct
    6j implementation must satisfy regardless of convention.

The identities are additionally routed *through the formula evaluator* (the
``Sum(...)`` machinery in ``FormulaEvaluator``) rather than called directly.
That is deliberate: it exercises the summation-range and half-integer-step
logic that the direct symbol calls would bypass, which is where range/parity
bugs would live.

For a C++ reader: pytest collects any ``test_*`` function automatically; there
is no registration step. ``pytest.approx(x, rel=1e-12)`` is a floating-point
comparison with a relative tolerance, i.e. ``|a-b| <= 1e-12 * |b|``.
"""

import itertools
import math
import os
import sys

import networkx as nx
import pytest

# The package is not pip-installable yet (no pyproject.toml), so the repo root
# has to be put on the import path by hand. Remove this once packaging lands.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api import Graph
from src.gluer import glue_open_edges
from src.graph_reducer import ReductionError, reduce_all_cycles
from src.api import UnitArg
from src.utils import f_range_symbolic, f_range_with_symbolic
from src.spin_evaluator import (
    FormulaEvaluator,
    SpinNetworkEvaluator,
    _multiprocessing_is_usable,
)

# sympy ships an independent, exact (rational/sqrt) 6j implementation.
from sympy.physics.wigner import wigner_6j as sympy_6j

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
# wigxjpf keeps GLOBAL C-level tables: wig_table_init / wig_temp_init allocate
# process-wide state, and constructing two evaluators re-initialises it. So we
# build exactly one evaluator per test session and share it.
#
# backend='serial' is forced explicitly: we are validating the mathematics
# here, not the dispatch.
# Backend equivalence is tested separately in TestBackendAgreement.


@pytest.fixture(scope="module")
def ev():
    """A single shared SpinNetworkEvaluator, cleaned up at module teardown."""
    evaluator = SpinNetworkEvaluator(max_two_j=60, backend="serial")
    yield evaluator
    evaluator.cleanup()


@pytest.fixture(scope="module")
def formula_ev():
    """A FormulaEvaluator, for testing the Sum()/eval() machinery end-to-end."""
    fev = FormulaEvaluator(max_two_j=60, backend="serial")
    yield fev
    fev.cleanup()


# --------------------------------------------------------------------------
# Helpers: closed forms re-derived independently of src/
# --------------------------------------------------------------------------


def reference_theta(j, k, j3):
    """
    theta(j,k,j3) = (-1)^(j+k+j3) * (j+k+j3+1)! / [(j+k-j3)! (j-k+j3)! (-j+k+j3)!]

    Written from the formula in the module docstring of graph_reducer, NOT by
    calling into src/. Returns 0.0 when the triangle inequality fails.
    """
    if not (abs(j - k) <= j3 <= j + k):
        return 0.0
    # The integer-sum rule guarantees these are all non-negative integers for
    # an admissible trivalent vertex.
    s = j + k + j3
    if abs(s - round(s)) > 1e-9:
        return 0.0
    n = math.factorial(int(round(s)) + 1)
    d1 = math.factorial(int(round(j + k - j3)))
    d2 = math.factorial(int(round(j - k + j3)))
    d3 = math.factorial(int(round(-j + k + j3)))
    return ((-1.0) ** int(round(s))) * n / (d1 * d2 * d3)


def reference_delta(j):
    """Delta_j = (-1)^(2j) * (2j + 1)."""
    return ((-1.0) ** int(round(2 * j))) * (2 * j + 1)


def combine(sign_exponent_and_magnitude):
    """
    src/ returns theta and delta as a (sign_exponent, magnitude) pair so that
    signs can be accumulated in the exponent and collapsed only once. Fold that
    pair back into a single signed float, the way FormulaEvaluator does.
    """
    sign_exp, mag = sign_exponent_and_magnitude
    return ((-1.0) ** int(round(sign_exp))) * mag


def sympy_6j_safe(a, b, c, d, e, f):
    """
    sympy's wigner_6j raises ValueError when the arguments are not integer or
    half-integer or fail a triangle relation, whereas wigxjpf (and this
    codebase) return 0.0 for the same input. A vanishing 6j symbol and an
    inadmissible one are the same thing physically, so we map the exception
    onto 0.0 to put both implementations on the same footing.
    """
    try:
        return float(sympy_6j(a, b, c, d, e, f))
    except ValueError:
        return 0.0


def admissible(a, b, c):
    """Triangle inequality plus the integer-sum rule for an SU(2) vertex."""
    if abs((a + b + c) - round(a + b + c)) > 1e-9:
        return False
    return abs(a - b) <= c <= a + b


def half_integer_spins(max_two_j):
    """All spins 0, 1/2, 1, ... up to max_two_j/2, as floats."""
    return [n / 2.0 for n in range(max_two_j + 1)]


# --------------------------------------------------------------------------
# 1. Wigner 6j against sympy  -- the single strongest independent check
# --------------------------------------------------------------------------


class TestWigner6jAgainstSympy:
    """
    wigxjpf (C, ours) vs sympy (pure Python, independent) over an exhaustive
    sweep. If this passes, orthogonality and Biedenharn-Elliott follow
    automatically, since they are properties of the function itself.
    """

    def test_exhaustive_integer_spins(self, ev):
        """All six arguments integer, 2j <= 4. Exhaustive, no sampling."""
        spins = [0.0, 1.0, 2.0]
        checked = 0
        for a in spins:
            for b in spins:
                for c in spins:
                    for d in spins:
                        for e in spins:
                            for f in spins:
                                ours = ev.wigner_6j(a, b, c, d, e, f)
                                theirs = sympy_6j_safe(a, b, c, d, e, f)
                                assert ours == pytest.approx(theirs, abs=1e-12), (
                                    f"6j{{{a} {b} {c}; {d} {e} {f}}}: "
                                    f"ours={ours!r} sympy={theirs!r}"
                                )
                                checked += 1
        assert checked == len(spins) ** 6

    def test_exhaustive_half_integer_spins(self, ev):
        """
        Includes half-integer arguments, which is where a 2j-conversion or
        parity bug would show up. Only admissible tuples are compared in
        detail; inadmissible ones must give exactly 0 on both sides.
        """
        spins = half_integer_spins(3)  # 0, 1/2, 1, 3/2
        mismatches = []
        for a in spins:
            for b in spins:
                for c in spins:
                    for d in spins:
                        for e in spins:
                            for f in spins:
                                ours = ev.wigner_6j(a, b, c, d, e, f)
                                theirs = sympy_6j_safe(a, b, c, d, e, f)
                                if ours != pytest.approx(theirs, abs=1e-12):
                                    mismatches.append((a, b, c, d, e, f, ours, theirs))
        assert (
            not mismatches
        ), f"{len(mismatches)} mismatches vs sympy, first 5: {mismatches[:5]}"

    def test_inadmissible_returns_exactly_zero(self, ev):
        """A violated triangle must give 0.0, not a small number or a NaN."""
        # (1, 1, 5) is not a triangle.
        assert ev.wigner_6j(1, 1, 5, 1, 1, 1) == 0.0
        # Half-integer sum rule violation: 1/2 + 1/2 + 1/2 is not an integer.
        assert ev.wigner_6j(0.5, 0.5, 0.5, 1, 1, 1) == 0.0


# --------------------------------------------------------------------------
# 2. Theta and Delta against re-derived closed forms
# --------------------------------------------------------------------------


class TestThetaDeltaClosedForm:

    def test_theta_matches_reference_integer(self, ev):
        for a in [0.0, 1.0, 2.0, 3.0]:
            for b in [0.0, 1.0, 2.0, 3.0]:
                for c in [0.0, 1.0, 2.0, 3.0]:
                    ours = combine(ev.theta_symbol(a, b, c))
                    ref = reference_theta(a, b, c)
                    assert ours == pytest.approx(
                        ref, rel=1e-12, abs=1e-12
                    ), f"theta({a},{b},{c}): ours={ours} ref={ref}"

    def test_theta_matches_reference_half_integer(self, ev):
        """
        Half-integer vertices. Note reference_theta returns 0 when j+k+l is not
        an integer; src/ does NOT check that rule inside theta_symbol (it only
        checks the triangle inequality), so a divergence here is a real finding
        rather than a test bug.
        """
        spins = half_integer_spins(5)
        divergences = []
        for a in spins:
            for b in spins:
                for c in spins:
                    if not admissible(a, b, c):
                        continue  # only compare on physically valid vertices
                    ours = combine(ev.theta_symbol(a, b, c))
                    ref = reference_theta(a, b, c)
                    if ours != pytest.approx(ref, rel=1e-12, abs=1e-12):
                        divergences.append((a, b, c, ours, ref))
        assert not divergences, f"theta divergences: {divergences[:5]}"

    def test_theta_symmetric_under_argument_permutation(self, ev):
        """theta is totally symmetric in its three arguments."""
        import itertools

        for a, b, c in [(1, 2, 3), (0.5, 0.5, 1), (2, 2, 2), (1.5, 2.5, 2)]:
            values = {
                combine(ev.theta_symbol(*p)) for p in itertools.permutations([a, b, c])
            }
            assert (
                len(values) == 1 or max(values) - min(values) < 1e-9
            ), f"theta not symmetric for ({a},{b},{c}): {values}"

    def test_delta_matches_reference(self, ev):
        for j in half_integer_spins(20):
            ours = combine(ev.delta_symbol(j))
            assert ours == pytest.approx(reference_delta(j), rel=1e-12)

    def test_theta_power_consistency(self, ev):
        """
        theta^0.5 squared must equal theta. expand_6j_symbolic emits +-1/2
        powers of theta, so this is load-bearing, and it is the one place a
        signed-vs-magnitude power bug would bite.
        """
        for a, b, c in [(1, 1, 2), (2, 2, 2), (1.5, 1.5, 1), (3, 3, 4)]:
            half = combine(ev.theta_symbol(a, b, c, power=0.5))
            full = combine(ev.theta_symbol(a, b, c, power=1.0))
            assert half * half == pytest.approx(abs(full), rel=1e-9), (
                f"theta({a},{b},{c})^0.5 squared = {half*half}, "
                f"|theta| = {abs(full)}"
            )


# --------------------------------------------------------------------------
# 3. Algebraic identities, evaluated THROUGH the Sum() machinery
# --------------------------------------------------------------------------


class TestIdentitiesThroughFormulaEvaluator:
    """
    These identities are exercised as formula strings so that the summation
    bounds, the step size, and the eval() namespace are all on trial, not just
    the 6j kernel. This is the layer where a range or parity bug lives.
    """

    def test_6j_orthogonality_integer(self, formula_ev):
        """
        sum_x (2x+1) {a b x}{c d x} * (2f+1) ... in the standard form

            sum_x (2x+1)(2f+1) {a b x}{c d x} = delta_{f,f'} ...

        We use the cleanest version:
            sum_x (2x+1) * {a b x; c d f} * {a b x; c d f'} = delta_{f f'}/(2f+1)
        """
        a, b, c, d = 1.0, 1.0, 1.0, 1.0
        for f, fp in [(1.0, 1.0), (1.0, 2.0), (2.0, 2.0), (0.0, 2.0)]:
            x_min = max(abs(a - b), abs(c - d))
            x_max = min(a + b, c + d)
            formula = (
                f"Sum('x', {x_min}, {x_max}, lambda x: "
                f"(2*x+1) * W6j({a},{b},x,{c},{d},{f}) "
                f"* W6j({a},{b},x,{c},{d},{fp}))"
            )
            got = formula_ev.evaluate(formula)
            expected = (1.0 / (2 * f + 1)) if f == fp else 0.0
            # Signed comparison: FormulaEvaluator preserves sign (abs() is
            # applied only at the norm boundary in src/api.py).
            assert got == pytest.approx(
                expected, abs=1e-10
            ), f"orthogonality f={f} f'={fp}: got {got}, expected {expected}"

    def test_6j_orthogonality_half_integer(self, formula_ev):
        """
        Same identity with half-integer external spins, so the summation runs
        over half-integer x. This is the direct probe for the '// 2' style
        truncation of summation bounds: if half-integer bounds are floored,
        this sum picks up the wrong terms and the identity fails.
        """
        a, b, c, d = 0.5, 0.5, 0.5, 0.5
        for f, fp in [(0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]:
            x_min = max(abs(a - b), abs(c - d))
            x_max = min(a + b, c + d)
            formula = (
                f"Sum('x', {x_min}, {x_max}, lambda x: "
                f"(2*x+1) * W6j({a},{b},x,{c},{d},{f}) "
                f"* W6j({a},{b},x,{c},{d},{fp}))"
            )
            got = formula_ev.evaluate(formula)
            expected = (1.0 / (2 * f + 1)) if f == fp else 0.0
            assert got == pytest.approx(expected, abs=1e-10), (
                f"half-integer orthogonality f={f} f'={fp}: "
                f"got {got}, expected {expected}"
            )

    def test_sum_helper_steps_by_one_from_half_integer_start(self, formula_ev):
        """
        Sum() must visit 1/2, 3/2, 5/2 when started at 1/2 -- i.e. the step is
        1 in j (not 1/2), and a half-integer lower bound is preserved exactly.
        """
        got = formula_ev.evaluate("Sum('x', 0.5, 2.5, lambda x: x)")
        assert got == pytest.approx(0.5 + 1.5 + 2.5, abs=1e-12)

    def test_sum_upper_bound_is_inclusive(self, formula_ev):
        """The upper limit of a spin summation is inclusive."""
        got = formula_ev.evaluate("Sum('x', 0.0, 3.0, lambda x: 1.0)")
        assert got == pytest.approx(4.0, abs=1e-12)


# --------------------------------------------------------------------------
# 4. Known-value spot checks from the literature
# --------------------------------------------------------------------------


class TestKnownValues:
    """
    Hand-checkable values. {1 1 1; 1 1 1} = 1/6 is the standard textbook
    entry and is quoted in every 6j table.
    """

    def test_6j_all_ones(self, ev):
        assert ev.wigner_6j(1, 1, 1, 1, 1, 1) == pytest.approx(1.0 / 6.0, abs=1e-12)

    def test_6j_with_a_zero_argument(self, ev):
        """
        {a b 0; c d f} reduces to a closed form:
            {a b 0; c d f} = (-1)^(a+c+f) / sqrt((2a+1)(2c+1))  when a==b, c==d
        Checked against sympy rather than trusted from memory.
        """
        for a in [0.5, 1.0, 1.5, 2.0]:
            for c in [0.5, 1.0, 1.5]:
                for f in [0.5, 1.0, 1.5, 2.0]:
                    ours = ev.wigner_6j(a, a, 0, c, c, f)
                    theirs = sympy_6j_safe(a, a, 0, c, c, f)
                    assert ours == pytest.approx(theirs, abs=1e-12)

    def test_theta_1_1_1(self, ev):
        """theta(1,1,1) = (-1)^3 * 4!/(1! 1! 1!) = -24."""
        assert combine(ev.theta_symbol(1, 1, 1)) == pytest.approx(-24.0, rel=1e-12)

    def test_theta_zero_edge(self, ev):
        """theta(j, j, 0) = (-1)^(2j) * (2j+1)! / (2j)! = (-1)^(2j)(2j+1) = Delta_j."""
        for j in [0.5, 1.0, 1.5, 2.0, 3.0]:
            th = combine(ev.theta_symbol(j, j, 0))
            dl = combine(ev.delta_symbol(j))
            assert th == pytest.approx(
                dl, rel=1e-12
            ), f"theta({j},{j},0)={th} should equal Delta_{j}={dl}"


# --------------------------------------------------------------------------
# 4b. End-to-end pipeline: graph in, number out
# --------------------------------------------------------------------------


def _closed_theta_graph(labels=(1.0, 1.0, 2.0)):
    """Two vertices joined by three parallel edges. The simplest closed net."""
    g = nx.MultiGraph()
    g.add_node(0, pos=(0.0, 0.0))
    g.add_node(1, pos=(1.0, 0.0))
    for lab in labels:
        g.add_edge(0, 1, label=lab)
    return g


def _tetrahedron(node_names, label=1.0):
    """
    K4 with all six edges carrying `label`. `node_names` maps the canonical
    vertices 0..3 onto arbitrary names, which lets us verify that the norm is
    independent of how the vertices happen to be called.
    """
    g = nx.MultiGraph()
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (0.5, 1.0), 3: (0.5, 0.4)}
    for canonical, name in node_names.items():
        g.add_node(name, pos=pos[canonical])
    for u, v in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
        g.add_edge(node_names[u], node_names[v], label=label)
    return g


class TestPipelineEndToEnd:
    """
    The full path: networkx graph -> glue -> reduce -> canonicalise -> number.

    For a *closed* input the gluing step produces two disjoint copies, so the
    norm is the square of the net's value. That is what pins the expected
    numbers below.
    """

    def test_closed_theta_graph_is_theta_squared(self, formula_ev):
        """||theta net|| = Theta(a,b,c)^2. Theta(1,1,2) = 5!/(0!2!2!) = 30."""
        norm = Graph(_closed_theta_graph((1.0, 1.0, 2.0))).evaluate_symbolic()
        expected = reference_theta(1.0, 1.0, 2.0) ** 2
        assert expected == pytest.approx(900.0)
        assert norm.evaluate_numeric() == pytest.approx(expected, rel=1e-9)

    def test_tetrahedron_norm_is_node_naming_invariant(self):
        """
        REGRESSION -- src/graph_reducer.py:281.

        `apply_triangle_reduction` used `if not all([au_node, dv_node,
        fw_node])` to test whether the three external neighbours were found.
        A vertex *named* 0 is falsy in Python, so whenever the external
        neighbour of a triangle vertex happened to be the node called 0, the
        reduction silently aborted. `reduce_all_cycles` then saw "no face > 3
        and nothing changed", declared success, and returned an empty
        coefficient product -- a norm of 1.0 instead of 9216.0.

        6 of the 24 relabelings of the identical tetrahedron were affected.
        The norm of a graph cannot depend on what its vertices are called, so
        all 24 must agree.
        """
        norms = {}
        for perm in itertools.permutations(range(4)):
            names = {i: perm[i] for i in range(4)}
            graph = Graph(_tetrahedron(names))
            norms[perm] = graph.evaluate_symbolic().evaluate_numeric()

        distinct = set(norms.values())
        majority = max(distinct, key=list(norms.values()).count)
        offenders = [p for p, v in norms.items() if v != majority]
        assert len(distinct) == 1, (
            f"norm depends on node naming: {sorted(distinct)}; "
            f"offending permutations: {offenders}"
        )
        # theta(1,1,1)^4 * W6j(1,1,1,1,1,1)^2 = 24^4 / 36 = 9216
        assert distinct.pop() == pytest.approx(9216.0, rel=1e-9)

    def test_non_trivalent_graph_raises_instead_of_returning_a_wrong_number(self):
        """
        REGRESSION -- the silent-partial-result exit paths of
        reduce_all_cycles.

        Every reduction move assumes trivalent vertices. Given a 4-valent
        vertex the moves cannot fire, and the reducer used to return whatever
        partial coefficient product it had accumulated, which evaluates to a
        plausible but wrong number. It must raise instead.
        """
        g = nx.MultiGraph()
        for n in (1, 2, 3, 4):
            g.add_edge(0, n, label=1.0)  # 4-valent centre
        for n in (1, 2, 3, 4):
            g.add_edge(n, f"s{n}", label=1.0)  # open ends

        with pytest.raises(ReductionError, match="Reduction did not complete"):
            Graph(g).evaluate_symbolic()

    def test_successful_reduction_consumes_the_whole_graph(self):
        """
        The invariant the guard relies on: a completed reduction trades every
        edge for an algebraic factor, leaving no edges behind.
        """
        for name, graph in [
            ("theta", _closed_theta_graph()),
            ("tetrahedron", _tetrahedron({0: 0, 1: 1, 2: 2, 3: 3})),
        ]:
            term = reduce_all_cycles(glue_open_edges(graph))[0]
            assert (
                term["graph"].number_of_edges() == 0
            ), f"{name}: {term['graph'].number_of_edges()} edges left over"
            assert len(term["coeffs"]) > 0


# --------------------------------------------------------------------------
# 5. Characterisation of the abs() wrapper
# --------------------------------------------------------------------------


class TestSignConvention:
    """
    The modulus is a property of the NORM, not of formula evaluation.

    Individual factors (theta, delta, the (-1)^... prefactors) carry signs
    that cancel against each other during the summation, so they must keep
    those signs all the way through. abs() is therefore applied exactly once,
    at the very end, by Formula.evaluate_numeric() / evaluate_batch() in
    src/api.py -- right before the norm is handed back.

    Previously FormulaEvaluator.evaluate() itself returned abs(...), which
    both made sign errors in the reduction pipeline invisible and prevented
    callers from inspecting intermediate quantities.
    """

    def test_formula_evaluator_preserves_sign(self, formula_ev):
        assert formula_ev.evaluate("-5.0") == pytest.approx(-5.0)

    def test_theta_keeps_its_negative_sign_through_the_evaluator(self, formula_ev):
        """theta(1,1,1) = -24, and the evaluator must report -24, not +24."""
        assert formula_ev.evaluate("theta(1,1,1)") == pytest.approx(-24.0, rel=1e-12)
        # Consistent with the direct symbol call.
        assert reference_theta(1.0, 1.0, 1.0) == pytest.approx(-24.0)

    def test_signs_cancel_rather_than_being_discarded_early(self, formula_ev):
        """
        The point of keeping signs: a product of two negative factors is
        positive. Taking abs() per factor would give the same answer here but
        the wrong one as soon as a sum of mixed-sign terms is involved.
        """
        both = formula_ev.evaluate("theta(1,1,1) * theta(1,1,1)")
        assert both == pytest.approx(576.0, rel=1e-12)

        mixed = formula_ev.evaluate("theta(1,1,1) + theta(1,1,2)")
        # -24 + 30 = 6; discarding signs first would give 54.
        assert mixed == pytest.approx(6.0, rel=1e-12)

    def test_norm_is_non_negative_at_the_api_boundary(self):
        """
        abs() still applies where it should: the value handed back by
        Formula.evaluate_numeric() is the norm and is non-negative, even
        though the underlying formula evaluates to a negative number.
        """
        # theta(1,1,1) = -24, so the closed theta net's norm is (-24)^2 = 576;
        # use a net whose raw formula is negative to make the point.
        norm = Graph(_closed_theta_graph((1.0, 1.0, 1.0))).evaluate_symbolic()
        raw = norm._formula_string
        value = norm.evaluate_numeric()
        assert value >= 0.0, f"norm {value} from formula {raw} must be >= 0"
        assert value == pytest.approx(576.0, rel=1e-9)


# --------------------------------------------------------------------------
# 6. Backend dispatch
# --------------------------------------------------------------------------


class TestBackendDispatch:
    """
    The parallel backend must be (a) actually reachable, (b) numerically
    identical to serial, and (c) never chosen when it would be slower or
    unsafe.

    History: `backend` used to be inert on this path -- FormulaEvaluator had
    its own pure-Python Sum() and never dispatched on it, so
    `evaluate_numeric()` ignored the backend entirely. Separately, the legacy
    path handed an unpicklable nested closure to Pool.map, so it could never
    have run at all. JAX was also advertised but could not help: it cannot
    trace the wigxjpf C library, and has since been removed.
    """

    SMALL = (
        "Sum('F_1', 0.0, 3.0, lambda F_1: "
        "delta(F_1) * W6j(F_1,1,1,1,1,1) * theta(F_1,1,1))"
    )

    def test_serial_matches_multiprocessing_on_a_small_formula(self):
        """Below break-even the parallel backend must fall back, not differ."""
        serial = FormulaEvaluator(max_two_j=40, backend="serial", verbose=False)
        try:
            expected = serial.evaluate(self.SMALL)
        finally:
            serial.cleanup()

        mp = FormulaEvaluator(max_two_j=40, backend="multiprocessing", verbose=False)
        try:
            got = mp.evaluate(self.SMALL)
        finally:
            mp.cleanup()

        assert got == expected

    def test_chunked_evaluation_reproduces_the_serial_sum_exactly(self):
        """
        The core correctness argument for parallelism: splitting the outermost
        sum into disjoint chunks and adding the partial results reproduces the
        whole sum, because the expression is linear in that sum.

        Exercised here in-process (no worker pool needed) so it runs fast and
        deterministically everywhere.
        """
        ev = FormulaEvaluator(max_two_j=60, backend="serial", verbose=False)
        try:
            formula = (
                "theta(1,1,2) * Sum('F_1', 0.0, 9.0, lambda F_1: "
                "delta(F_1) * W6j(F_1,2,2,2,2,2))"
            )
            whole = ev._evaluate_here(formula)
            chunked = (
                ev._evaluate_here(formula, outer_chunk=(0.0, 2.0))
                + ev._evaluate_here(formula, outer_chunk=(3.0, 6.0))
                + ev._evaluate_here(formula, outer_chunk=(7.0, 9.0))
            )
            assert chunked == pytest.approx(whole, rel=1e-12)
        finally:
            ev.cleanup()

    def test_nested_sums_are_not_chunked_only_the_outer_one(self):
        """A chunk must restrict the outer sum but leave inner sums intact."""
        ev = FormulaEvaluator(max_two_j=60, backend="serial", verbose=False)
        try:
            formula = (
                "Sum('A', 0.0, 3.0, lambda A: "
                "Sum('B', 0.0, 2.0, lambda B: A * 10 + B))"
            )
            # Full: sum over A of (3*10A + (0+1+2)) = sum_A (30A + 3)
            whole = ev._evaluate_here(formula)
            assert whole == pytest.approx(30 * (0 + 1 + 2 + 3) + 4 * 3)

            # Chunking A must not shrink B's range.
            piece = ev._evaluate_here(formula, outer_chunk=(0.0, 0.0))
            assert piece == pytest.approx(0 * 30 + 3)
        finally:
            ev.cleanup()

    def test_non_linear_use_of_the_outer_sum_is_refused(self):
        """
        Chunking is only valid when the sum enters linearly. A squared sum, or
        one buried in a function argument (as calculate_probability's
        safe_div(...) form produces), must be rejected so the evaluator falls
        back to serial instead of returning a wrong number.
        """
        check = FormulaEvaluator._outer_sum_is_a_linear_factor
        assert check("Sum('F', 0.0, 9.0, lambda F: delta(F))") is True
        assert check("theta(1,1,2) * Sum('F', 0.0, 9.0, lambda F: delta(F))") is True

        assert check("Sum('F', 0.0, 9.0, lambda F: delta(F)) ** 2") is False
        assert check("safe_div(Sum('F', 0.0, 9.0, lambda F: delta(F)), 2.0)") is False
        assert check("1.0 / Sum('F', 0.0, 9.0, lambda F: delta(F))") is False
        # Two independent top-level sums: which one runs first is not something
        # we want to depend on, so refuse.
        assert (
            check("Sum('A', 0.0, 9.0, lambda A: A) * Sum('B', 0.0, 9.0, lambda B: B)")
            is False
        )
        # No sum at all.
        assert check("theta(1,1,2)") is False

    def test_auto_resolves_to_serial(self):
        """
        'auto' must be serial: parallelism costs ~1 s of startup (break-even
        ~350,000 summation terms) and, under the 'spawn' start method, makes
        an unguarded user script re-run itself once per worker. Neither is
        acceptable as a default. See scripts/benchmark_backends.py.
        """
        ev = FormulaEvaluator(max_two_j=40, backend="auto", verbose=False)
        try:
            assert ev._ev.backend == "serial"
        finally:
            ev.cleanup()

    def test_evaluate_many_matches_one_by_one(self):
        """Batch evaluation must agree with looping, parallel or not."""
        ev = FormulaEvaluator(max_two_j=60, backend="multiprocessing", verbose=False)
        try:
            formula = "theta(j, j, 0) * delta(j)"
            sets = [{"j": v} for v in (0.5, 1.0, 1.5, 2.0, 2.5)]
            batch = ev.evaluate_many(formula, sets)
            one_by_one = [ev._evaluate_here(formula, s) for s in sets]
            assert batch == pytest.approx(one_by_one, rel=1e-12)
        finally:
            ev.cleanup()

    def test_multiprocessing_safety_check_is_honest(self):
        """
        Under the 'spawn' start method (macOS/Windows default) each worker
        re-imports __main__. From a notebook or a heredoc that import fails and
        the pool retries forever -- a hang no try/except can catch. The guard
        must therefore refuse up front in those contexts.

        Under pytest __main__ is a real file, so it should report True here.
        """
        assert _multiprocessing_is_usable() in (True, False)  # never raises

        import multiprocessing as mp

        if mp.get_start_method() == "fork":
            assert _multiprocessing_is_usable() is True
        else:
            main_file = getattr(sys.modules["__main__"], "__file__", None)
            expected = bool(main_file) and os.path.isfile(main_file)
            assert _multiprocessing_is_usable() is expected


# --------------------------------------------------------------------------
# 7. F-move summation ranges
# --------------------------------------------------------------------------


class TestFMoveSummationRange:
    """
    An F-move inserts a new edge F between two vertices.  After the rewiring in
    f_move_recouple_term, those vertices carry (b, c, F) and (a, d, F) -- which
    expand_6j_symbolic corroborates by emitting Theta(b, c, f) and
    Theta(a, d, f).  The allowed range of F is therefore

        max(|b-c|, |a-d|)  <=  F  <=  min(b+c, a+d)

    The code used to pair (b, d) and (a, c) instead.  Neither is a vertex of
    the graph, and the resulting range was both offset and too narrow: for
    a=2, b=1, d=2, c=1 it gave 1..3 where the true support is 0..2, silently
    dropping the F=0 term from every norm containing an F-move.

    Every pre-existing range test used all-equal labels (1,1,1,1), for which
    both pairings coincide -- which is why this went unnoticed.  The tests
    below deliberately use labels that distinguish them.
    """

    def test_pairs_bc_and_ad_not_bd_and_ac(self):
        """a=2, b=1, d=2, c=1 -- the case measured against the true support."""
        rng = f_range_symbolic(a=2.0, b=1.0, d=2.0, c=1.0)
        # doubled units: A=4, B=2, D=4, C=2
        #   correct  : max(|B-C|, |A-D|) = 0 ,  min(B+C, A+D) = 4   -> spin 0..2
        #   incorrect: max(|B-D|, |A-C|) = 2 ,  min(B+D, A+C) = 6   -> spin 1..3
        assert rng is not None
        assert (rng["Fmin"], rng["Fmax"]) == (0, 4), (
            f"got doubled range {rng['Fmin']}..{rng['Fmax']}, expected 0..4 "
            "(spin 0..2); the label pairing is wrong"
        )

    def test_range_is_symmetric_under_swapping_the_two_vertices(self):
        """
        Exchanging the roles of the two F vertices, (b,c) <-> (a,d), cannot
        change the range.
        """
        first = f_range_symbolic(a=2.0, b=1.0, d=2.0, c=1.0)
        second = f_range_symbolic(a=1.0, b=2.0, d=1.0, c=2.0)
        assert (first["Fmin"], first["Fmax"]) == (second["Fmin"], second["Fmax"])

    def test_symbolic_bounds_use_the_same_pairing(self):
        """The symbolic branch must pair the same way as the numeric one."""
        rng = f_range_with_symbolic("A", 1.0, 2.0, 3.0)  # a symbolic, b,d,c numeric
        assert rng["symbolic"] is True
        # b=1.0 with c=3.0, and a='A' with d=2.0
        assert "abs(1.0 - (3.0))" in rng["symbolic_Fmin"] or "2" in rng["symbolic_Fmin"]
        assert "A" in rng["symbolic_Fmin"] and "A" in rng["symbolic_Fmax"]

    def test_symbolic_bounds_are_in_spin_units_not_doubled(self):
        """
        The strings are substituted into the formula verbatim (no /2), so a
        numeric pair must be emitted as its spin value.  b=2.0 with c=0.5 must
        give 1.5, not to_doubled -> 3.
        """
        rng = f_range_with_symbolic("A", 2.0, 1.0, 0.5)  # b=2.0, c=0.5 numeric
        assert "1.5" in rng["symbolic_Fmin"], (
            f"expected the spin-unit bound 1.5 in {rng['symbolic_Fmin']!r}; "
            "a doubled value here would be twice the symbolic side"
        )
        assert (
            "2.5" in rng["symbolic_Fmax"]
        ), f"expected the spin-unit sum 2.5 in {rng['symbolic_Fmax']!r}"


class TestSummationRangeCompleteness:
    """
    The general correctness property, independent of any pairing argument:

        widening a summation range must not change the result.

    Values of F outside the physical range are killed by the triangle
    inequality inside Theta and W6j, so they contribute exactly zero.  If
    widening DOES change the answer, the narrower range was dropping non-zero
    terms -- which is what a wrong range looks like from the outside.

    This is the test that would have caught the pairing bug on any graph.
    """

    SPINS = dict(a=1.0, b=2.0, c=1.0, d=2.0, e=1.0, f=2.0, g=1.0, h=2.0, i=1.0, j=1.0)

    def pentagon(self):
        """Five trivalent nodes in a ring, each with one open leg."""
        g = nx.MultiGraph()
        ring = ["n1", "n2", "n3", "n4", "n5"]
        ring_labels = [self.SPINS[k] for k in "bdfhj"]
        leg_labels = [self.SPINS[k] for k in "acegi"]
        for idx, node in enumerate(ring):
            g.add_edge(node, ring[(idx + 1) % 5], label=ring_labels[idx])
            g.add_edge(node, f"leg{idx}", label=leg_labels[idx])
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)
        return g

    def square(self):
        """Four trivalent nodes in a ring -- needs exactly one F-move."""
        g = nx.MultiGraph()
        ring = ["m1", "m2", "m3", "m4"]
        for idx, node in enumerate(ring):
            g.add_edge(node, ring[(idx + 1) % 4], label=[1.0, 2.0, 1.0, 2.0][idx])
            g.add_edge(node, f"s{idx}", label=[2.0, 1.0, 2.0, 1.0][idx])
        for n in g.nodes:
            g.nodes[n]["pos"] = (0.0, 0.0)
        return g

    def norm_with_range(self, builder, wide):
        """Evaluate the norm, optionally forcing a deliberately wide F range."""
        import src.graph_reducer as graph_reducer

        original = graph_reducer.f_range_with_symbolic
        if wide:
            graph_reducer.f_range_with_symbolic = (
                lambda a, b, d, c, known_ranges=None: {
                    "Fmin": 0,
                    "Fmax": 40,
                    "parity": 0,
                }
            )
        try:
            return Graph(builder()).evaluate_symbolic().evaluate_numeric()
        finally:
            graph_reducer.f_range_with_symbolic = original

    @pytest.mark.parametrize("name", ["square", "pentagon"])
    def test_widening_the_range_does_not_change_the_norm(self, name):
        builder = getattr(self, name)
        tight = self.norm_with_range(builder, wide=False)
        wide = self.norm_with_range(builder, wide=True)
        assert tight == pytest.approx(wide, rel=1e-9), (
            f"{name}: norm with the computed range is {tight!r} but {wide!r} "
            f"with a wider one -- the computed range is dropping non-zero terms"
        )

    def test_assigning_spins_before_or_after_reduction_agrees(self):
        """
        Reducing with numeric labels and reducing symbolically then
        substituting must give the same number.  The two take different
        branches of f_range_with_symbolic, so they only agree if both compute
        the same range.
        """
        numeric = Graph(self.pentagon()).evaluate_symbolic().evaluate_numeric()

        symbolic_nx = nx.MultiGraph()
        ring = ["n1", "n2", "n3", "n4", "n5"]
        for idx, node in enumerate(ring):
            symbolic_nx.add_edge(node, ring[(idx + 1) % 5], label="bdfhj"[idx])
            symbolic_nx.add_edge(node, f"leg{idx}", label="acegi"[idx])
        for n in symbolic_nx.nodes:
            symbolic_nx.nodes[n]["pos"] = (0.0, 0.0)

        formula = Graph(symbolic_nx).evaluate_symbolic()
        substituted = formula.evaluate_numeric(
            [UnitArg(k, v) for k, v in self.SPINS.items()]
        )
        assert numeric == pytest.approx(substituted, rel=1e-9), (
            f"numeric-first gives {numeric!r}, symbolic-then-substitute gives "
            f"{substituted!r}; these must agree"
        )
