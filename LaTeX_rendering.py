import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import symbols, latex
from IPython.display import display, Math

# Enable LaTeX rendering in Matplotlib
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def latex_6j(a, b, f, c, d, e):
    # Full LaTeX environment works with usetex=True
    return r"\begin{Bmatrix} " + f"{a} & {b} & {f} \\\\ {c} & {d} & {e}" + r" \end{Bmatrix}"

def latex_formatting(terms):
    """
    Prepare LaTeX-formatted requests and term mapping from the given terms.
    Each term consists of a graph and its associated coefficients.
    """
    ORDER = {
        "6j": 0,
        "theta": 1,
        "big delta": 2,
        "big delta inverse": 3,
        "Kronecker delta": 4,
    }

    factors = []  # <-- move factors out of loop so we can return all of them

    for term in terms:
        coeffs = term.get("coeffs", [])

        # sort coeff list by type priority
        coeffs = sorted(
            coeffs,
            key=lambda c: ORDER.get(c.get("type"), 999)
        )

        for c in coeffs:
            typ = c.get("type")
            fixed = c.get("fixed", {})

            if typ == "6j":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                d = fixed.get("d", fixed.get("D"))
                cc = fixed.get("c", fixed.get("C"))
                e = fixed.get("e", fixed.get("E"))

                rng = c.get("sum_range2")
                if rng:
                    fidx = c.get("sum_index", "f")
                    fmin = rng["Fmin"] / 2
                    fmax = rng["Fmax"] / 2
                    # Proper sum limits: \sum_{f=...}^{...}
                    factors.append(
                        rf"\sum_{{{fidx}={fmin}}}^{{{fmax}}} " + latex_6j(a, b, fidx, cc, d, e)
                    )
                else:
                    f = fixed.get("f", fixed.get("F"))
                    factors.append(latex_6j(a, b, f, cc, d, e))

            elif typ == "theta":
                a = fixed.get("a", fixed.get("A"))
                b = fixed.get("b", fixed.get("B"))
                cval = fixed.get("c", fixed.get("C"))
                factors.append(rf"\Theta\left({a},{b},{cval}\right)")

            elif typ == "big delta":
                j = fixed.get("j", fixed.get("J"))
                factors.append(rf"\Delta_{{{j}}}")

            elif typ == "big delta inverse":
                j = fixed.get("j", fixed.get("J"))
                factors.append(rf"\frac{{1}}{{\Delta_{{{j}}}}}")

            elif typ == "Kronecker delta":
                c1 = fixed.get("c", fixed.get("C"))
                d1 = fixed.get("d", fixed.get("D"))
                factors.append(rf"\delta_{{{c1},{d1}}}")

            else:
                factors.append(str(c))

    return factors

def render_latex_expression(terms):
    """
    Render a list of factors (strings) as a LaTeX-style expression.
    Factors should already be formatted like:
        6j(a,b,f,c,d,e), θ(a,b,c), δ(c,d), etc.
    """
    factors = latex_formatting(terms)
    expr_str = r" \cdot ".join(factors)

    # Display in notebook using MathJax (works with full LaTeX)
    display(Math(expr_str))

    # Render as figure using matplotlib usetex
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${expr_str}$", ha='center', va='center', fontsize=16)
    plt.show()

def save_latex_pdf(terms, filename="expression.pdf"):
    factors = latex_formatting(terms)
    expr_str = r" \cdot ".join(factors)

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${expr_str}$", ha='center', va='center', fontsize=16)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LaTeX expression to {filename}")