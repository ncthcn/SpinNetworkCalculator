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
    Canonical key for a W6j coefficient.
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

    a, b, e, c1, d, f = args
    mats = [
        (a, b, e, c1, d, f),
        (b, c1, a, e, f, d),
        (c1, a, b, f, d, e),
        (d, e, f, a, b, c1),
        (e, f, d, b, c1, a),
        (f, d, e, c1, a, b),
    ]
    mats2 = mats + [(x[3], x[4], x[5], x[0], x[1], x[2]) for x in mats]
    return ("W6j", min(mats2), power)

def canonicalise_terms(terms):
    """
    Canonicalise a list of terms.
    Handles both dict and tuple coefficients.
    Returns new list of canonicalised terms.
    """
    canon_terms = []

    for term in terms:
        counter = {}

        for c in term["coeffs"]:
            typ = c[0] if isinstance(c, tuple) else c.get("type")
            if typ == "theta":
                key = theta_key(c)
            elif typ == "W6j":
                key = wigner_sixj_key(c)
            else:
                # keep other coefficients as-is
                key = (typ, str(c))
            counter[key] = counter.get(key, 0) + 1

        # reconstruct canonical coeffs
        new_coeffs = []
        for k, count in counter.items():
            typ = k[0]
            args = k[1]
            power = k[2] if len(k) > 2 else count
            if typ == "theta":
                new_coeffs.append({"type": "theta", "args": args, "power": power})
            elif typ == "W6j":
                new_coeffs.append({"type": "W6j", "args": args, "power": power})
            else:
                new_coeffs.append(args)  # fallback

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


