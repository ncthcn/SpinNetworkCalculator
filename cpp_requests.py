

def collect_requests_with_indices(terms):
    """
    Collect backend evaluation requests while explicitly keeping track
    of which evaluations depend on which summed indices.
    """

    sixj = {}
    theta = {}
    delta = {}

    for tidx, term in enumerate(terms):
        for coeff in term["coeffs"]:

            typ = coeff["type"]
            args = tuple(coeff["args2"])

            if typ == "6j":
                idx = coeff.get("sum_index", None)
                sixj.setdefault(args, set()).add(idx)

            elif typ == "theta":
                theta.setdefault(args, set()).add(None)

            elif typ == "delta":
                j = coeff["fixed"]["J"]
                delta.setdefault(j, set()).add(None)

    return sixj, theta, delta
