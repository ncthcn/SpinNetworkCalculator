import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import symbols, latex
from IPython.display import display, Math

# Enable LaTeX rendering in Matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def extract_args(c):
    """
    Given a coefficient (dict or tuple), return a,b,c,... args and power.
    """
    if isinstance(c, dict):
        typ = c.get("type")
        power = c.get("power", 1)
        fixed = c.get("fixed", {})
        # choose either numeric keys or original keys
        args = tuple(fixed.get(k.upper(), fixed.get(k)) for k in c.get("list_order", fixed.keys()))
        return typ, args, power

    elif isinstance(c, tuple):
        # tuple from expand_6j_symbolic: ("theta", (a,b,c), power)
        if len(c) == 3:
            return c[0], c[1], c[2]
        elif len(c) == 2:
            return c[0], c[1], 1
        else:
            return "unknown", c, 1
    else:
        # fallback
        return "unknown", (c,), 1


def latex_6j(a, b, f, c, d, e):
    # Full LaTeX environment works with usetex=True
    return r"\begin{Bmatrix} " + f"{a} & {b} & {f} \\\\ {c} & {d} & {e}" + r" \end{Bmatrix}"

# def latex_formatting(terms):
#     """
#     Prepare LaTeX-formatted requests and term mapping from the given terms.
#     Each term consists of a graph and its associated coefficients.
#     """
#     ORDER = {
#         "sum": 0,
#         "sign": 1,
#         "W6j": 2,
#         "6j": 3,
#         "theta": 4,
#         "delta": 5,
#         "delta inverse": 5,
#         "Kronecker": 6,
#     }

#     for term in terms:
#         coeffs = term.get("coeffs", [])

#         # sort coeff list by type priority
#         coeffs = sorted(
#             coeffs,
#             key=lambda c: ORDER.get(c.get("type"), 999)
#         )

#         factors = []

#         for c in coeffs:
#             typ = c.get("type")
#             fixed = c.get("fixed", {})

#             if typ == "sum":
#                 f = c.get("index", "f")
#                 rng = c.get("range2")
#                 if rng:
#                     fmin = rng["Fmin"] / 2
#                     fmax = rng["Fmax"] / 2
#                     factors.append(
#                         rf"\sum_{{{f}={fmin}}}^{{{fmax}}} "
#                     )
#                 else:
#                     factors.append(f"\sum_{{{f}}}")

#             elif typ == "sign":
#                 args = fixed.get("args", [])
#                 arg_strs = []
#                 for c, val in args:
#                     if c is not None: 
#                         arg_strs.append(f"{c}{val}")
#                     else:
#                         arg_strs.append(f"+{val}")
#                 factors.append(rf"(-1)^{{{' '.join(arg_strs)}}}")            

#             elif typ == "6j":
#                 a = fixed.get("a", fixed.get("A"))
#                 b = fixed.get("b", fixed.get("B"))
#                 f = fixed.get("f", fixed.get("F"))
#                 d = fixed.get("d", fixed.get("D"))
#                 c = fixed.get("c", fixed.get("C"))
#                 e = fixed.get("e", fixed.get("E"))
#                 power = fixed.get("power", 1)
#                 if power != 1:
#                     factors.append(rf"\left{latex_6j(a,b,f,c,d,e)}\right^{{{power}}}")
#                 else:
#                     factors.append(latex_6j(a,b,f,c,d,e))

#             elif typ == "W6j":
#                 a = fixed.get("a", fixed.get("A"))
#                 b = fixed.get("b", fixed.get("B"))
#                 f = fixed.get("f", fixed.get("F"))
#                 d = fixed.get("d", fixed.get("D"))
#                 c = fixed.get("c", fixed.get("C"))
#                 e = fixed.get("e", fixed.get("E"))
#                 power = fixed.get("power", 1)
#                 if power != 1:
#                     factors.append(rf"\left{latex_6j(a,b,e,c,d,f)}\right^{{{power}}}_{{W}}")
#                 else:
#                     factors.append(rf"\left{latex_6j(a,b,e,c,d,f)}\right_{{W}}")

#             elif typ == "theta":
#                 a = fixed.get("a", fixed.get("A"))
#                 b = fixed.get("b", fixed.get("B"))
#                 cval = fixed.get("c", fixed.get("C"))
#                 power = fixed.get("power", 1)
#                 if power != 1:
#                     factors.append(rf"\Theta\left({a},{b},{cval}\right)^{{{power}}}")
#                 else:
#                     factors.append(rf"\Theta\left({a},{b},{cval}\right)")

#             elif typ == "delta":
#                 j = fixed.get("j", fixed.get("J"))
#                 power = fixed.get("power", 1)
#                 if power != 1:
#                     factors.append(rf"\Delta_{{{j}}}^{{{power}}}")
#                 else:
#                     factors.append(rf"\Delta_{{{j}}}")

#             elif typ == "Kronecker":
#                 c1 = fixed.get("c", fixed.get("C"))
#                 d1 = fixed.get("d", fixed.get("D"))
#                 factors.append(rf"\delta_{{{c1},{d1}}}")

#             else:
#                 factors.append(str(c))

#     return factors

# def save_latex_pdf(terms, filename="expression.pdf"):
#     factors = latex_formatting(terms)
#     expr_str = r" \cdot ".join(factors)

#     fig, ax = plt.subplots(figsize=(8, 1))
#     ax.axis('off')
#     ax.text(0.5, 0.5, f"${expr_str}$", ha='center', va='center', fontsize=16)
#     fig.savefig(filename, bbox_inches='tight')
#     plt.close(fig)
#     print(f"Saved LaTeX expression to {filename}")




def latex_formatting(terms):
    ORDER = {
        "sum": 0,
        "sign": 1,
        "W6j": 2,
        "6j": 3,
        "theta": 4,
        "delta": 5,
        "delta inverse": 5,
        "Kronecker": 6,
    }

    all_factors = []

    for term in terms:
        coeffs = term.get("coeffs", [])

        # sort coeff list by type priority
        coeffs = sorted(
            coeffs,
            key=lambda c: ORDER.get(c.get("type") if isinstance(c, dict) else "", 999)
        )

        factors = []

        for c in coeffs:
            if not isinstance(c, dict):
                # skip invalid entries
                continue

            typ = c.get("type", "")
            fixed = c.get("fixed", {})

            if typ == "sum":
                f = c.get("index", "f")
                rng = c.get("range2")
                if rng:
                    fmin = rng["Fmin"] / 2
                    fmax = rng["Fmax"] / 2
                    factors.append(rf"\sum_{{{f}={fmin}}}^{{{fmax}}} ")
                else:
                    factors.append(rf"\sum_{{{f}}}")

            elif typ == "sign":
                args = fixed.get("args", [])
                arg_strs = []
                for sgn, val in args:
                    if sgn is not None: 
                        arg_strs.append(f"{sgn}{val}")
                    else:
                        arg_strs.append(f"+{val}")
                factors.append(rf"(-1)^{{{' '.join(arg_strs)}}}")

            elif typ in ("6j", "W6j"):
                # prefer args tuple, else get from fixed
                args = c.get("args")
                if not args:
                    args = (
                        fixed.get("a", fixed.get("A")),
                        fixed.get("b", fixed.get("B")),
                        fixed.get("f", fixed.get("F")),
                        fixed.get("c", fixed.get("C")),
                        fixed.get("d", fixed.get("D")),
                        fixed.get("e", fixed.get("E")),
                    )
                a, b, fval, cval, d, e = args
                power = fixed.get("power", 1)
                if typ == "6j":
                    if power != 1:
                        factors.append(rf"\left{latex_6j(a,b,fval,cval,d,e)}\right)^{{{power}}}")
                    else:
                        factors.append(latex_6j(a,b,fval,cval,d,e))
                else:  # W6j
                    if power != 1:
                        factors.append(rf"\left{latex_6j(a,b,fval,cval,d,e)}\right^{{{power}}}_{{{W}}}")
                    else:
                        factors.append(rf"\left{latex_6j(a,b,fval,cval,d,e)}\right_{{{W}}}")

            elif typ == "theta":
                args = c.get("args")
                if not args:
                    args = (
                        fixed.get("a", fixed.get("A")),
                        fixed.get("b", fixed.get("B")),
                        fixed.get("c", fixed.get("C"))
                    )
                a, b, cval = args
                power = fixed.get("power", 1)
                if power != 1:
                    factors.append(rf"\Theta\left({a},{b},{cval}\right)^{{{power}}}")
                else:
                    factors.append(rf"\Theta\left({a},{b},{cval}\right)")

            elif typ == "delta":
                j = fixed.get("j", fixed.get("J"))
                power = fixed.get("power", 1)
                if typ == "delta inverse":
                    if power != 1:
                        factors.append(rf"\Delta_{{{j}}}^{{-{power}}}")
                    else:
                        factors.append(rf"\Delta_{{{j}}}^{{-1}}")
                else:
                    if power != 1:
                        factors.append(rf"\Delta_{{{j}}}^{{{power}}}")
                    else:
                        factors.append(rf"\Delta_{{{j}}}")

            elif typ == "Kronecker":
                c1 = fixed.get("c", fixed.get("C"))
                d1 = fixed.get("d", fixed.get("D"))
                factors.append(rf"\delta_{{{c1},{d1}}}")

        all_factors.append(factors)

    return all_factors


# def latex_formatting(terms):
#     """
#     Prepare LaTeX-formatted requests and term mapping from the given terms.
#     Works with both dict and tuple coefficients.
#     """
#     ORDER = {
#         "sum": 0,
#         "sign": 1,
#         "W6j": 2,
#         "6j": 3,
#         "theta": 4,
#         "delta": 5,
#         "delta inverse": 5,
#         "Kronecker": 6,
#     }

#     formatted_terms = []

#     for term in terms:
#         coeffs = term.get("coeffs", []) if isinstance(term, dict) else term
#         # sort safely: if c is tuple, assign a default type for ordering
#         coeffs_sorted = sorted(
#             coeffs,
#             key=lambda c: ORDER.get(c["type"], 999) if isinstance(c, dict) else 999
#         )

#         factors = []

#         for c in coeffs_sorted:
#             if isinstance(c, dict):
#                 typ = c.get("type")
#                 fixed = c.get("fixed", {})
#             elif isinstance(c, tuple):
#                 typ = c[0]
#                 # keep tuple contents for formatting
#                 fixed = c[1] if len(c) > 1 else {}
#             else:
#                 # fallback: convert to string
#                 factors.append(str(c))
#                 continue

#             if typ == "sum":
#                 f = fixed.get("index", "f") if isinstance(fixed, dict) else fixed
#                 rng = fixed.get("range2") if isinstance(fixed, dict) else None
#                 if rng:
#                     fmin = rng["Fmin"] / 2
#                     fmax = rng["Fmax"] / 2
#                     factors.append(rf"\sum_{{{f}={fmin}}}^{{{fmax}}}")
#                 else:
#                     factors.append(rf"\sum_{{{f}}}")

#             elif typ == "sign":
#                 args = fixed.get("args", []) if isinstance(fixed, dict) else fixed
#                 arg_strs = []
#                 for c2, val in args:
#                     if c2 is not None:
#                         arg_strs.append(f"{c2}{val}")
#                     else:
#                         arg_strs.append(f"+{val}")
#                 factors.append(rf"(-1)^{{{' '.join(arg_strs)}}}")

#             elif typ in ("6j", "W6j"):
#                 if isinstance(fixed, dict):
#                     a = fixed.get("a", fixed.get("A"))
#                     b = fixed.get("b", fixed.get("B"))
#                     f = fixed.get("f", fixed.get("F"))
#                     d = fixed.get("d", fixed.get("D"))
#                     cval = fixed.get("c", fixed.get("C"))
#                     e = fixed.get("e", fixed.get("E"))
#                     power = fixed.get("power", 1)
#                 else:
#                     a, b, f, cval, d, e = fixed
#                     power = 1

#                 latex_term = latex_6j(a, b, f, cval, d, e)
#                 if power != 1:
#                     factors.append(rf"\left{latex_term}\right^{{{power}}}")
#                 else:
#                     factors.append(latex_term)

#             elif typ == "theta":
#                 if isinstance(fixed, dict):
#                     a = fixed.get("a", fixed.get("A"))
#                     b = fixed.get("b", fixed.get("B"))
#                     cval = fixed.get("c", fixed.get("C"))
#                     power = fixed.get("power", 1)
#                 else:
#                     a, b, cval = fixed
#                     power = 1
#                 if power != 1:
#                     factors.append(rf"\Theta\left({a},{b},{cval}\right)^{{{power}}}")
#                 else:
#                     factors.append(rf"\Theta\left({a},{b},{cval}\right)")

#             elif typ == "delta":
#                 j = fixed.get("j", fixed.get("J")) if isinstance(fixed, dict) else fixed
#                 power = fixed.get("power", 1) if isinstance(fixed, dict) else 1
#                 if power != 1:
#                     factors.append(rf"\Delta_{{{j}}}^{{{power}}}")
#                 else:
#                     factors.append(rf"\Delta_{{{j}}}")

#             elif typ == "Kronecker":
#                 if isinstance(fixed, dict):
#                     c1 = fixed.get("c", fixed.get("C"))
#                     d1 = fixed.get("d", fixed.get("D"))
#                 else:
#                     c1, d1 = fixed
#                 factors.append(rf"\delta_{{{c1},{d1}}}")

#             else:
#                 factors.append(str(c))

#         formatted_terms.append(factors)

#     # flatten list of lists into single list of strings
#     return [item for sublist in formatted_terms for item in sublist]



def save_latex_pdf(terms, filename="expression.pdf"):
    factors = latex_formatting(terms)
    # expr_str = r" \cdot ".join(factors)
    # flatten all factor lists into a single list of strings
    flat_factors = [f for sublist in factors for f in sublist]
    expr_str = r" \cdot ".join(flat_factors)



    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${expr_str}$", ha='center', va='center', fontsize=16)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LaTeX expression to {filename}")