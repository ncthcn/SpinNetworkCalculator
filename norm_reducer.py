from utils import is_numeric_label

#-------------------------
# New coefficient builders
#-------------------------

def build_sign_coeff(args):
    """
    Build a sign coefficient dict.
    """
    fixed = {"args": args}
    return {
        "type": "sign",
        "fixed": fixed
    }

def build_wigner_6j_coeff(a, b, e, c, d, f, power=1):
    fixed = {"a": a, "b": b, "e": e, "c": c, "d": d, "f": f}

    # doubled numeric values
    for key in ["a", "b", "c", "d", "e", "f"]:
        val = fixed[key]
        if is_numeric_label(val):
            fixed[key.upper()] = to_doubled(val)
    print(f"Built 6j coeff with fixed labels:", a, b, e, c, d, f)

    return {
        "type": "W6j",
        "list_order": ["a", "b", "e", "c", "d", "f"],
        "fixed": fixed,
        "power": power
    }

def apply_kroneckers(term):
    """
    Take a term {'sign', 'coeffs'} and:
    1. Apply all Kronecker deltas (numeric or symbolic).
    2. Merge summed indices; intersect ranges when merged.
    3. Substitute merged indices in 6j, theta, delta, and sum coefficients.
    4. Remove zero terms.
    Returns modified term or None if term vanishes.
    """
    equalities = {}  # label -> set of equivalent labels
    sums = {}        # sum index -> sum dict

    # 1) Gather all Kronecker deltas
    for c in term["coeffs"]:
        if c["type"] == "Kronecker":
            x, y = c["fixed"]["c"], c["fixed"]["d"]

            # numeric contradiction → term vanishes
            if is_numeric_label(x) and is_numeric_label(y) and x != y:
                return None

            # symbolic → record equivalence
            equalities.setdefault(x, set()).add(y)
            equalities.setdefault(y, set()).add(x)

    # 2) Flatten equivalence sets into representative mapping
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

    # 3) Apply substitution to all coefficients & collect sums
    new_coeffs = []
    for c in term["coeffs"]:
        if c["type"] == "Kronecker":
            continue  # already processed

        nc = c.copy()

        # Substitute in args
        if "args" in nc:
            nc["args"] = tuple(subst.get(a, a) for a in nc["args"])

        # Substitute in fixed dict
        if "fixed" in nc:
            nc["fixed"] = {k: subst.get(v, v) for k, v in nc["fixed"].items()}

        # Merge sums
        if nc.get("type") == "sum":
            idx = nc["index"]
            rng = nc.get("range2")
            if idx in sums:
                old_rng = sums[idx].get("range2")
                if old_rng and rng:
                    # intersect ranges
                    Fmin = max(old_rng["Fmin"], rng["Fmin"])
                    Fmax = min(old_rng["Fmax"], rng["Fmax"])
                    parity = old_rng.get("parity", rng.get("parity"))
                    if Fmin > Fmax:
                        return None  # empty sum → term vanishes
                    sums[idx]["range2"] = {"Fmin": Fmin, "Fmax": Fmax, "parity": parity}
                elif rng:
                    sums[idx]["range2"] = rng
            else:
                sums[idx] = nc
        else:
            new_coeffs.append(nc)

    # 4) Append merged sums
    new_coeffs.extend(sums.values())

    # 5) Propagate substitutions in all nested sums inside 6j/theta/delta
    for c in new_coeffs:
        if c.get("type") in ("6j", "theta", "delta", "delta inverse"):
            # If the coeff contains a sum index, apply substitution
            if "sum_index" in c and c["sum_index"] in subst:
                old_idx = c["sum_index"]
                new_idx = subst[old_idx]
                c["sum_index"] = new_idx
                if "gen_args" in c and callable(c["gen_args"]):
                    # you may regenerate the args if needed
                    c["args"] = tuple(new_idx if x == old_idx else x for x in c["args"])

    term["coeffs"] = new_coeffs
    return term


# def expand_6j_symbolic(coeff):
#     """
#     Replace one 6j entry with the symbolic internal 6j definition.

#     Input coeff:
#         { "type":"6j", "args":(a, b, f, c, d, e) }
#     """
#     fixed = coeff.get("fixed", {})

#     a = fixed.get("a", fixed.get("A"))
#     b = fixed.get("b", fixed.get("B"))
#     d = fixed.get("d", fixed.get("D"))
#     c = fixed.get("c", fixed.get("C"))
#     e = fixed.get("e", fixed.get("E"))
#     f = fixed.get("f", fixed.get("F"))

#     out = []

#     # sign contribution
#     out.append(("sign", [("-",a), ("-",b), ("-",c), ("-",d), ("2", f)]))

#     # Δ contributions
#     out.append(("delta", f))
    
#     # θ contributions
#     out.append(("theta", (a, b, e), .5))
#     out.append(("theta", (c, d, e), .5))
#     out.append(("theta", (b, c, f), -.5))
#     out.append(("theta", (a, d, f), -.5))

#     # Wigner 6j contribution
#     out.append(("W6j", (a, b, e, c, d, f), 1))

#     return out

def expand_6j_symbolic(coeff):
    fixed = coeff.get("fixed", {})
    a = fixed.get("a", fixed.get("A"))
    b = fixed.get("b", fixed.get("B"))
    d = fixed.get("d", fixed.get("D"))
    c = fixed.get("c", fixed.get("C"))
    e = fixed.get("e", fixed.get("E"))
    f = fixed.get("f", fixed.get("F"))

    out = []

    # sign contribution
    out.append({"type": "sign", "fixed": {"args": [("-",a), ("-",b), ("-",c), ("-",d), ("2", f)]}, "power": 1})

    # delta contribution
    out.append({"type": "delta", "fixed": {"j": f}, "power": 1})

    # theta contributions
    out.append({"type": "theta", "args": (a, b, e), "power": 0.5})
    out.append({"type": "theta", "args": (c, d, e), "power": 0.5})
    out.append({"type": "theta", "args": (b, c, f), "power": -0.5})
    out.append({"type": "theta", "args": (a, d, f), "power": -0.5})

    # Wigner 6j contribution
    out.append({"type": "W6j", "fixed": {"a": a, "b": b, "e": e, "c": c, "d": d, "f": f}, "power": 1})

    return out

# -------------------------
# Key functions for canonicalisation
# -------------------------

def canonicalise_sign(sign_coeff):
    """
    Canonicalise a sign coefficient by:
    1. Summing all numeric terms
    2. Collecting variable terms with their coefficients

    Input: sign_coeff with args = [(coeff, val), ...]
    Output: canonicalised sign_coeff with combined terms
    """
    args = sign_coeff.get("fixed", {}).get("args", [])

    # Accumulate numeric sum and variable terms
    numeric_sum = 0
    var_terms = {}  # variable -> coefficient

    for coeff_str, val in args:
        # Parse coefficient (could be '-', '+', '2', etc.)
        if coeff_str == '-':
            coeff = -1
        elif coeff_str == '+' or coeff_str is None:
            coeff = 1
        else:
            try:
                coeff = float(coeff_str)
            except:
                coeff = 1  # fallback

        # Check if val is numeric or variable
        if isinstance(val, (int, float)):
            numeric_sum += coeff * val
        else:
            # Variable term
            var_name = str(val)
            var_terms[var_name] = var_terms.get(var_name, 0) + coeff

    # If only numeric (no variables), simplify based on parity
    if not var_terms or all(c == 0 for c in var_terms.values()):
        # Check if numeric_sum is even or odd
        if numeric_sum % 2 == 0:
            return None  # (-1)^even = 1, remove the sign
        else:
            # (-1)^odd = -1, return a special marker
            return {"type": "sign_value", "value": -1}

    # Rebuild args list for mixed numeric/variable case
    new_args = []

    # Add numeric sum if non-zero
    if numeric_sum != 0:
        if numeric_sum > 0:
            new_args.append(('+', numeric_sum))
        else:
            new_args.append(('-', -numeric_sum))

    # Add variable terms sorted alphabetically
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

    return {
        "type": "sign",
        "fixed": {"args": new_args}
    }

def sort_mixed_args(args):
    """
    Sort arguments for canonical form.
    Numbers first (ints/floats), then symbols (strings).
    """
    def key(x):
        return (0, x) if isinstance(x, (int, float)) else (1, str(x))
    return tuple(sorted(args, key=key))

# def theta_key(c):
#     """
#     Canonical key for a theta coefficient.
#     Accepts dict with 'args' or 'fixed'->'args', or raw tuple/list.
#     """
#     if isinstance(c, dict):
#         args = c.get("args") or c.get("fixed", {}).get("args")
#         if args is None:
#             raise ValueError(f"No args found in coefficient: {c}")
#     elif isinstance(c, (tuple, list)):
#         args = tuple(c)
#     else:
#         raise TypeError(f"Unexpected type for theta_key: {type(c)}")

#     return ("theta", sort_mixed_args(args))

def theta_key(c):
    """
    Canonical key for a theta coefficient.
    Works with either:
      - dict: {"type":"theta", "args":(...)} or {"fixed": {...}}
      - tuple: ("theta", args, power)
    Returns a sorted tuple of arguments and power.
    """
    # extract args and power
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

    # custom sort to mix numbers and strings
    def sort_mixed(x):
        return (0, x) if isinstance(x, (int, float)) else (1, str(x))

    sorted_args = tuple(sorted(args, key=sort_mixed))
    return ("theta", sorted_args, power)

def wigner_sixj_key(c):
    """
    Canonical key for a W6j coefficient using correct Regge symmetries.

    The Wigner 6j symbol {a b c; d e f} has 24 symmetries (tetrahedral group).
    These come from:
    - Even permutations of the 3 columns (6 permutations)
    - Interchanging upper and lower rows in each column (×2 for each column)
    - Combined: 6 × 4 = 24 total symmetries

    Accepts either:
      - dict with 'fixed'
      - tuple: ("W6j", args, power)
    Returns lexicographically minimal Regge-equivalent tuple plus power.
    """
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

    # Generate all 24 Regge symmetries
    # Start with 6 even column permutations, then apply row swaps
    column_perms = [
        (a, b, c, d, e, f),  # identity
        (b, c, a, e, f, d),  # cycle (123)
        (c, a, b, f, d, e),  # cycle (132)
        (a, c, b, d, f, e),  # swap columns 2↔3
        (c, b, a, f, e, d),  # swap columns 1↔3
        (b, a, c, e, d, f),  # swap columns 1↔2
    ]

    # For each column permutation, generate 4 variants by swapping rows in pairs
    all_symmetries = []
    for (j1, j2, j3, j4, j5, j6) in column_perms:
        # Original
        all_symmetries.append((j1, j2, j3, j4, j5, j6))
        # Swap rows in columns 1&2
        all_symmetries.append((j4, j5, j3, j1, j2, j6))
        # Swap rows in columns 1&3
        all_symmetries.append((j4, j2, j6, j1, j5, j3))
        # Swap rows in columns 2&3
        all_symmetries.append((j1, j5, j6, j4, j2, j3))

    # Custom key for sorting mixed types (numbers first, then strings)
    def sort_key(mat):
        return tuple((0, x) if isinstance(x, (int, float)) else (1, str(x)) for x in mat)

    return ("W6j", min(all_symmetries, key=sort_key), power)

def canonicalise_terms(terms):
    """
    Canonicalise a list of terms.
    Handles both dict and tuple coefficients.
    Returns new list of canonicalised terms.
    """
    canon_terms = []

    for term in terms:
        power_accumulator = {}  # maps (type, args) -> total_power
        sign_coeffs = []  # collect all sign coefficients to merge
        non_canonical_coeffs = []  # for coeffs that shouldn't be canonicalized

        for c in term["coeffs"]:
            typ = c[0] if isinstance(c, tuple) else c.get("type")
            if typ == "theta":
                key = theta_key(c)  # returns ("theta", sorted_args, power)
                typ_key = (key[0], key[1])  # (type, args) without power
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + key[2]
            elif typ == "W6j":
                key = wigner_sixj_key(c)  # returns ("W6j", canonical_args, power)
                typ_key = (key[0], key[1])  # (type, args) without power
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + key[2]
            elif typ == "delta":
                # Accumulate delta powers by their index j
                fixed = c.get("fixed", {})
                j = fixed.get("j", fixed.get("J"))
                power = c.get("power", 1)
                typ_key = ("delta", j)
                power_accumulator[typ_key] = power_accumulator.get(typ_key, 0) + power
            elif typ == "sign":
                # Collect sign coefficients to merge
                sign_coeffs.append(c)
            else:
                # keep other coefficient types (sum, 6j, Kronecker) as-is
                non_canonical_coeffs.append(c)

        # Merge all sign coefficients into one
        overall_sign = 1  # Track the overall sign multiplier
        if sign_coeffs:
            # Merge all signs by combining their args
            merged_args = []
            for sign_c in sign_coeffs:
                args = sign_c.get("fixed", {}).get("args", [])
                merged_args.extend(args)

            # Create merged sign and canonicalize it
            merged_sign = {"type": "sign", "fixed": {"args": merged_args}}
            canonical_sign = canonicalise_sign(merged_sign)

            # Handle the canonicalized sign result
            if canonical_sign is None:
                # Sign simplified to 1, no need to add it
                sign_coeffs = []
            elif canonical_sign.get("type") == "sign_value":
                # Sign simplified to -1
                overall_sign = canonical_sign.get("value", 1)
                sign_coeffs = []
            elif canonical_sign.get("fixed", {}).get("args"):
                # Sign still has variable terms
                sign_coeffs = [canonical_sign]
            else:
                sign_coeffs = []

        # Reconstruct canonical coeffs
        new_coeffs = []

        # First, add the overall sign if it's -1
        if overall_sign == -1:
            new_coeffs.append({"type": "sign_value", "value": -1})

        # Collect sum indices to identify which coefficients depend on summed variables
        sum_indices = set()
        for c in non_canonical_coeffs:
            if isinstance(c, dict) and c.get("type") == "sum":
                sum_indices.add(c.get("index"))

        # Helper function to check if a coefficient contains a summed variable
        def contains_summed_var(coeff):
            if not isinstance(coeff, dict):
                return False

            # Check in args
            args = coeff.get("args", ())
            if any(str(arg) in sum_indices for arg in args):
                return True

            # Check in fixed
            fixed = coeff.get("fixed", {})
            for val in fixed.values():
                if isinstance(val, str) and val in sum_indices:
                    return True
                elif isinstance(val, dict):
                    for v in val.values():
                        if isinstance(v, str) and v in sum_indices:
                            return True
            return False

        # Separate coefficients into those without and with summed variables
        coeffs_before_sum = []
        coeffs_after_sum = []

        for (typ, args), total_power in power_accumulator.items():
            # Skip coefficients that cancel out (power = 0)
            if total_power == 0:
                continue

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

        # Build final coefficient list: sign → non-summed → sign(with vars) → sum → summed coeffs
        new_coeffs.extend(coeffs_before_sum)
        new_coeffs.extend(sign_coeffs)  # Variable-dependent signs go before sum
        new_coeffs.extend([c for c in non_canonical_coeffs if c.get("type") == "sum"])  # Sum
        new_coeffs.extend(coeffs_after_sum)
        new_coeffs.extend([c for c in non_canonical_coeffs if c.get("type") != "sum"])  # Other

        term["coeffs"] = new_coeffs
        canon_terms.append(term)

    return canon_terms

def reconstruct_terms_from_canonical(canon_terms):
    """
    Reconstruct terms in the original structure form (like reduce_all_cycles output)
    from the canonicalised terms.

    Input:
        canon_terms: list of canonicalised terms, each with 'coeffs' list
    Output:
        terms: list of terms with 'coeffs' list in the original format
    """
    reconstructed = []

    for term in canon_terms:
        coeffs = []
        for c in term.get("coeffs", []):
            # Only dicts have a "type"
            if isinstance(c, dict):
                coeffs.append(c)
            else:
                # Keep tuples or strings as-is
                coeffs.append(c)

        reconstructed.append({"coeffs": coeffs})

    return reconstructed


