from .utils import is_numeric_label, to_doubled

# -----------------------------------------------------------------------
# norm_reducer.py — post-reduction simplification and canonicalisation
# -----------------------------------------------------------------------
#
# After graph_reducer.py has finished, we have a term with a list of
# coefficient dicts. This module does three further passes:
#
#   1. apply_kroneckers  – substitute Kronecker-delta equalities so that
#                          redundant summation variables are merged.
#   2. expand_6j_symbolic – replace each standard 6j coefficient with its
#                           explicit definition in terms of theta/delta/W6j.
#   3. canonicalise_terms – merge identical factors (accumulate powers),
#                           sort arguments to a canonical form, and separate
#                           constant pre-factors from summation-dependent ones.

# -----------------------------------------------------------------------
# Coefficient builders (used only inside this module)
# -----------------------------------------------------------------------

# Creates a sign factor (-1)^(sum of args).
# 'args' is a list of (coefficient_string, variable_or_number) pairs,
# e.g. [('-', j1), ('+', j2), ('2', F_1)].
def build_sign_coeff(args):
    return {
        "type": "sign",
        "fixed": {"args": args}
    }

# -----------------------------------------------------------------------
# Kronecker substitution
# -----------------------------------------------------------------------

# Kronecker deltas arise whenever two edges that must carry the same spin
# are merged (degree-2 and 2-cycle reductions). This function:
#   1. Collects all Kronecker(x, y) constraints.
#   2. Builds a union-find over the constrained labels.
#   3. Substitutes every label with its representative.
#   4. Merges duplicate summation coefficients, intersecting their ranges.
# Returns the modified term, or None if a numeric contradiction is found
# (e.g. Kronecker(1, 2) forces j=1 AND j=2 simultaneously → term vanishes).
def apply_kroneckers(term):
    equalities = {}
    sums = {}

    for c in term["coeffs"]:
        if c["type"] == "Kronecker":
            x, y = c["fixed"]["c"], c["fixed"]["d"]

            if is_numeric_label(x) and is_numeric_label(y) and x != y:
                return None  # numeric contradiction: term is zero

            equalities.setdefault(x, set()).add(y)
            equalities.setdefault(y, set()).add(x)

    # Union-find: map every label to its canonical representative.
    mapping = {}

    def root(x):
        if x not in mapping:
            mapping[x] = x
        while mapping[x] != x:
            mapping[x] = mapping[mapping[x]]  # path compression
            x = mapping[x]
        return x

    for x, ys in equalities.items():
        base = root(x)
        for y in ys:
            mapping[root(y)] = base

    subst = {x: root(x) for x in mapping}

    new_coeffs = []
    for c in term["coeffs"]:
        if c["type"] == "Kronecker":
            continue  # already processed; no longer needed

        nc = c.copy()

        if "args" in nc:
            nc["args"] = tuple(subst.get(a, a) for a in nc["args"])

        if "fixed" in nc:
            nc["fixed"] = {k: subst.get(v, v) for k, v in nc["fixed"].items()}

        if nc.get("type") == "sum":
            idx = nc["index"]
            rng = nc.get("range2")
            if idx in sums:
                old_rng = sums[idx].get("range2")
                if old_rng and rng:
                    Fmin = max(old_rng["Fmin"], rng["Fmin"])
                    Fmax = min(old_rng["Fmax"], rng["Fmax"])
                    parity = old_rng.get("parity", rng.get("parity"))
                    if Fmin > Fmax:
                        return None  # empty summation range: term vanishes
                    sums[idx]["range2"] = {"Fmin": Fmin, "Fmax": Fmax, "parity": parity}
                elif rng:
                    sums[idx]["range2"] = rng
            else:
                sums[idx] = nc
        else:
            new_coeffs.append(nc)

    new_coeffs.extend(sums.values())

    # Propagate substitutions into nested sum indices inside other coefficients.
    for c in new_coeffs:
        if c.get("type") in ("6j", "theta", "delta", "delta inverse"):
            if "sum_index" in c and c["sum_index"] in subst:
                old_idx = c["sum_index"]
                new_idx = subst[old_idx]
                c["sum_index"] = new_idx
                if "gen_args" in c and callable(c["gen_args"]):
                    c["args"] = tuple(new_idx if x == old_idx else x for x in c["args"])

    term["coeffs"] = new_coeffs
    return term

# -----------------------------------------------------------------------
# 6j expansion
# -----------------------------------------------------------------------

# Replaces one standard 6j coefficient {a b f; c d e} with its explicit
# definition:
#   (-1)^(-a-b-c-d+2f)  ×  Δ_f  ×  Θ(a,b,e)^(1/2)  ×  Θ(c,d,e)^(1/2)
#   ×  Θ(b,c,f)^(-1/2)  ×  Θ(a,d,f)^(-1/2)  ×  W6j(a,b,e,c,d,f)
# The W6j symbol is the "pure" Wigner 6j, evaluated by wigxjpf.
# Returns a list of coefficient dicts that replace the single 6j entry.
def expand_6j_symbolic(coeff):
    fixed = coeff.get("fixed", {})
    a = fixed.get("a", fixed.get("A"))
    b = fixed.get("b", fixed.get("B"))
    d = fixed.get("d", fixed.get("D"))
    c = fixed.get("c", fixed.get("C"))
    e = fixed.get("e", fixed.get("E"))
    f = fixed.get("f", fixed.get("F"))

    out = []

    out.append({"type": "sign",
                "fixed": {"args": [("-", a), ("-", b), ("-", c), ("-", d), ("2", f)]},
                "power": 1})
    out.append({"type": "delta", "fixed": {"j": f}, "power": 1})
    out.append({"type": "theta", "args": (a, b, e), "power": 0.5})
    out.append({"type": "theta", "args": (c, d, e), "power": 0.5})
    out.append({"type": "theta", "args": (b, c, f), "power": -0.5})
    out.append({"type": "theta", "args": (a, d, f), "power": -0.5})
    out.append({"type": "W6j",
                "fixed": {"a": a, "b": b, "e": e, "c": c, "d": d, "f": f},
                "power": 1})

    return out

# -----------------------------------------------------------------------
# Canonicalisation helpers
# -----------------------------------------------------------------------

# Simplifies a sign coefficient (-1)^(sum) by:
#   - Summing all numeric contributions.
#   - Collecting variable contributions by name (to merge duplicates).
#   - Returning None if the exponent is always even (factor = 1).
#   - Returning {"type": "sign_value", "value": -1} if it's always odd.
def canonicalise_sign(sign_coeff):
    args = sign_coeff.get("fixed", {}).get("args", [])

    numeric_sum = 0
    var_terms = {}

    for coeff_str, val in args:
        if coeff_str == '-':
            coeff = -1
        elif coeff_str == '+' or coeff_str is None:
            coeff = 1
        else:
            try:
                coeff = float(coeff_str)
            except:
                coeff = 1

        if isinstance(val, (int, float)):
            numeric_sum += coeff * val
        else:
            var_name = str(val)
            var_terms[var_name] = var_terms.get(var_name, 0) + coeff

    if not var_terms or all(c == 0 for c in var_terms.values()):
        if numeric_sum % 2 == 0:
            return None         # (-1)^even = 1
        else:
            return {"type": "sign_value", "value": -1}

    new_args = []
    if numeric_sum != 0:
        new_args.append(('+', numeric_sum) if numeric_sum > 0 else ('-', -numeric_sum))

    for var in sorted(var_terms.keys()):
        coeff = var_terms[var]
        if coeff == 0:
            continue
        elif coeff == 1:
            new_args.append(('+', var))
        elif coeff == -1:
            new_args.append(('-', var))
        elif coeff > 0:
            new_args.append((f'+{coeff}', var))
        else:
            new_args.append((f'{coeff}', var))

    return {"type": "sign", "fixed": {"args": new_args}}

# Sorts a tuple of mixed numeric/symbolic arguments into canonical order
# (numbers first, then symbols alphabetically). Used by theta_key and
# wigner_sixj_key to make the key independent of argument order.
def sort_mixed_args(args):
    def key(x):
        return (0, x) if isinstance(x, (int, float)) else (1, str(x))
    return tuple(sorted(args, key=key))

# Canonical key for a theta coefficient.
# theta(a, b, c) = theta(b, a, c) = theta(c, a, b) … (symmetric in all 3
# arguments), so we sort them to produce a unique representative.
# Returns ("theta", sorted_args, power).
def theta_key(c):
    if isinstance(c, tuple):
        _, args, power = c
    elif isinstance(c, dict):
        args = c.get("args")
        if args is None:
            fixed = c.get("fixed", {})
            args = (fixed.get("a", fixed.get("A")),
                    fixed.get("b", fixed.get("B")),
                    fixed.get("c", fixed.get("C")))
        power = c.get("power", 1)
    else:
        raise TypeError(f"Unsupported theta coeff type: {type(c)}")

    def sort_mixed(x):
        return (0, x) if isinstance(x, (int, float)) else (1, str(x))

    sorted_args = tuple(sorted(args, key=sort_mixed))
    return ("theta", sorted_args, power)

# Canonical key for a Wigner 6j coefficient using its full 24-fold symmetry.
# The Wigner 6j symbol {a b c; d e f} is invariant under:
#   - Any permutation of its 3 columns.
#   - Swapping upper and lower entries in any pair of columns.
# Together this gives the tetrahedral group of order 24. We enumerate all 24
# equivalent forms and pick the lexicographically smallest one as the key.
# Returns ("W6j", canonical_args, power).
def wigner_sixj_key(c):
    if isinstance(c, tuple):
        _, args, power = c
    elif isinstance(c, dict):
        fixed = c.get("fixed", {})
        args = (fixed.get("a", fixed.get("A")),
                fixed.get("b", fixed.get("B")),
                fixed.get("e", fixed.get("E")),
                fixed.get("c", fixed.get("C")),
                fixed.get("d", fixed.get("D")),
                fixed.get("f", fixed.get("F")))
        power = c.get("power", 1)
    else:
        raise TypeError(f"Unsupported W6j coeff type: {type(c)}")

    a, b, c, d, e, f = args

    column_perms = [
        (a, b, c, d, e, f),
        (b, c, a, e, f, d),
        (c, a, b, f, d, e),
        (a, c, b, d, f, e),
        (c, b, a, f, e, d),
        (b, a, c, e, d, f),
    ]

    all_symmetries = []
    for (j1, j2, j3, j4, j5, j6) in column_perms:
        all_symmetries.append((j1, j2, j3, j4, j5, j6))
        all_symmetries.append((j4, j5, j3, j1, j2, j6))
        all_symmetries.append((j4, j2, j6, j1, j5, j3))
        all_symmetries.append((j1, j5, j6, j4, j2, j3))

    def sort_key(mat):
        return tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in mat)

    return ("W6j", min(all_symmetries, key=sort_key), power)

# -----------------------------------------------------------------------
# Canonicalisation pass
# -----------------------------------------------------------------------

# Combines identical factors by accumulating their powers (theta^p * theta^q
# → theta^(p+q)), merges all sign coefficients into one, and separates
# constant pre-factors from summation-dependent ones so the LaTeX output
# and the evaluator see a clean, well-ordered expression:
#
#   [constant thetas/deltas/sign]  ×  ∑_F [F-dependent thetas/W6j/…]
def canonicalise_terms(terms):
    canon_terms = []

    for term in terms:
        power_accumulator = {}
        sign_coeffs = []
        non_canonical_coeffs = []

        for c in term["coeffs"]:
            typ = c[0] if isinstance(c, tuple) else c.get("type")
            if typ == "theta":
                key = theta_key(c)
                typ_key = (key[0], key[1])
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + key[2]
            elif typ == "W6j":
                key = wigner_sixj_key(c)
                typ_key = (key[0], key[1])
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + key[2]
            elif typ == "delta":
                fixed = c.get("fixed", {})
                j = fixed.get("j", fixed.get("J"))
                power = c.get("power", 1)
                typ_key = ("delta", j)
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + power
            elif typ == "sign":
                sign_coeffs.append(c)
            else:
                non_canonical_coeffs.append(c)

        # Merge all sign factors into one simplified sign.
        overall_sign = 1
        if sign_coeffs:
            merged_args = []
            for sign_c in sign_coeffs:
                merged_args.extend(sign_c.get("fixed", {}).get("args", []))

            canonical_sign = canonicalise_sign({"type": "sign", "fixed": {"args": merged_args}})

            if canonical_sign is None:
                sign_coeffs = []
            elif canonical_sign.get("type") == "sign_value":
                overall_sign = canonical_sign.get("value", 1)
                sign_coeffs = []
            elif canonical_sign.get("fixed", {}).get("args"):
                sign_coeffs = [canonical_sign]
            else:
                sign_coeffs = []

        new_coeffs = []
        if overall_sign == -1:
            new_coeffs.append({"type": "sign_value", "value": -1})

        sum_indices = set(c.get("index") for c in non_canonical_coeffs
                         if isinstance(c, dict) and c.get("type") == "sum")

        # Helper: does this coefficient involve any summation variable?
        def contains_summed_var(coeff):
            if not isinstance(coeff, dict):
                return False
            args = coeff.get("args", ())
            if any(str(arg) in sum_indices for arg in args):
                return True
            fixed = coeff.get("fixed", {})
            for val in fixed.values():
                if isinstance(val, str) and val in sum_indices:
                    return True
                elif isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, str) and v in sum_indices:
                            return True
                elif isinstance(val, (list, tuple)):
                    for item in val:
                        if isinstance(item, (list, tuple)):
                            for sub_item in item:
                                if isinstance(sub_item, str) and sub_item in sum_indices:
                                    return True
                        elif isinstance(item, str) and item in sum_indices:
                            return True
            return False

        # Partition accumulated power factors into pre-sum and post-sum groups.
        coeffs_before_sum = []
        coeffs_after_sum = []

        for (typ, args), total_power in power_accumulator.items():
            if total_power == 0:
                continue  # factors that cancelled out

            if typ == "theta":
                coeff = {"type": "theta", "args": args, "power": total_power}
            elif typ == "W6j":
                coeff = {"type": "W6j", "args": args, "power": total_power}
            elif typ == "delta":
                coeff = {"type": "delta", "fixed": {"j": args}, "power": total_power}
            else:
                continue

            if contains_summed_var(coeff):
                coeffs_after_sum.append(coeff)
            else:
                coeffs_before_sum.append(coeff)

        # Final ordering: constant factors → sum symbol → F-dependent factors.
        new_coeffs.extend(coeffs_before_sum)

        sign_before_sum = [s for s in sign_coeffs if not contains_summed_var(s)]
        sign_after_sum  = [s for s in sign_coeffs if contains_summed_var(s)]

        new_coeffs.extend(sign_before_sum)
        new_coeffs.extend([c for c in non_canonical_coeffs if c.get("type") == "sum"])
        new_coeffs.extend(sign_after_sum)
        new_coeffs.extend(coeffs_after_sum)
        new_coeffs.extend([c for c in non_canonical_coeffs if c.get("type") != "sum"])

        term["coeffs"] = new_coeffs
        canon_terms.append(term)

    return canon_terms

# -----------------------------------------------------------------------
# Reconstruction
# -----------------------------------------------------------------------

# Converts canonical terms back into the standard term format
# (list of dicts with a "coeffs" key). This is a no-op for dict coefficients
# but handles any legacy tuple entries by passing them through unchanged.
# Used when we need to re-render the LaTeX after canonicalisation.
def reconstruct_terms_from_canonical(canon_terms):
    reconstructed = []
    for term in canon_terms:
        coeffs = [c for c in term.get("coeffs", [])]
        reconstructed.append({"coeffs": coeffs})
    return reconstructed
