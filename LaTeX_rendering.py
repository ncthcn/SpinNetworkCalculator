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

            elif typ == "6j":
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
                if power != 1:
                    factors.append(rf"\left{latex_6j(a,b,fval,cval,d,e)}\right)^{{{power}}}")
                else:
                    factors.append(latex_6j(a,b,fval,cval,d,e))
                
            elif typ == "W6j":
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
                a, b, e, cval, d, fval = args
                power = fixed.get("power", 1)
                factors.append(rf"\left{latex_6j(a,b,e,cval,d,fval)}\right^{{{power}}}_{{{W}}}")

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