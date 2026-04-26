import re
import matplotlib as mpl
import matplotlib.pyplot as plt
from sympy import symbols, latex
from IPython.display import display, Math

# Enable system LaTeX so that rendered PDFs use proper math fonts.
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def _sym_range_to_latex(expr: str) -> str:
    """Convert a symbolic range expression (Python syntax) to LaTeX.

    Examples::

        "max(abs(b - d), abs(a - c))"  ->  r"\\max(|b - d|, |a - c|)"
        "min(b + d, a + c)"            ->  r"\\min(b + d, a + c)"
        "abs(F_1 - 1.5)"               ->  r"|F_1 - 1.5|"
    """
    # abs(x) → |x|
    s = re.sub(r'abs\(([^)]+)\)', r'|\1|', expr)
    # max( → \max(,  min( → \min(
    s = s.replace('max(', r'\max(').replace('min(', r'\min(')
    # underscore in variable names (F_1 → F_{1})
    s = re.sub(r'([A-Za-z])_(\d+)', r'\1_{\2}', s)
    return s

# -----------------------------------------------------------------------
# LaTeX rendering of spin-network expressions
# -----------------------------------------------------------------------
#
# The evaluation pipeline produces a list of "term" dicts, each containing
# a "coeffs" list. These two functions turn that structure into a LaTeX
# string and save it as a PDF via matplotlib.

# Given a coefficient dict (or tuple), extracts the argument tuple and power.
# Both dict-style coefficients (from graph_reducer/norm_reducer) and legacy
# tuple-style coefficients are accepted so that this function can be called
# on raw or canonicalised terms indiscriminately.
def extract_args(c):
    if isinstance(c, dict):
        typ = c.get("type")
        power = c.get("power", 1)
        fixed = c.get("fixed", {})
        args = tuple(fixed.get(k.upper(), fixed.get(k)) for k in c.get("list_order", fixed.keys()))
        return typ, args, power
    elif isinstance(c, tuple):
        if len(c) == 3:
            return c[0], c[1], c[2]
        elif len(c) == 2:
            return c[0], c[1], 1
        else:
            return "unknown", c, 1
    else:
        return "unknown", (c,), 1


# Formats a standard Wigner 6j symbol as a LaTeX matrix block:
#   { a  b  f }
#   { c  d  e }
def latex_6j(a, b, f, c, d, e):
    return r"\{\begin{matrix} " + f"{a} & {b} & {f} \\\\ {c} & {d} & {e}" + r" \end{matrix}\}"


# Converts a list of terms into a nested list of LaTeX factor strings.
# Each inner list corresponds to one term; the factors are ready to be
# joined with \cdot for display.
#
# Coefficient types handled:
#   sum        →  \sum_{F_k=min}^{max}
#   sign_value →  -1  (only when value != 1)
#   sign       →  (-1)^{…}
#   6j         →  brace-matrix notation (standard, unexpanded)
#   W6j        →  brace-matrix notation with _W subscript
#   theta      →  \Theta(a, b, c)^p
#   delta      →  \Delta_j^p
#   Kronecker  →  \delta_{c,d}
def latex_formatting(terms):
    all_factors = []

    for term in terms:
        coeffs = term.get("coeffs", [])
        factors = []

        for c in coeffs:
            if not isinstance(c, dict):
                continue

            typ = c.get("type", "")
            fixed = c.get("fixed", {})

            if typ == "sum":
                f = c.get("index", "f")
                f_latex = re.sub(r'([A-Za-z])_(\d+)', r'\1_{\2}', f)
                rng = c.get("range2")
                if rng:
                    if rng.get("symbolic") and "symbolic_Fmin" in rng and "symbolic_Fmax" in rng:
                        fmin_s = _sym_range_to_latex(rng["symbolic_Fmin"])
                        fmax_s = _sym_range_to_latex(rng["symbolic_Fmax"])
                    else:
                        fmin_s = str(rng["Fmin"] / 2)
                        fmax_s = str(rng["Fmax"] / 2)
                    factors.append(rf"\sum_{{{f_latex}={fmin_s}}}^{{{fmax_s}}} ")
                else:
                    factors.append(rf"\sum_{{{f_latex}}}")

            elif typ == "sign_value":
                value = c.get("value", 1)
                if value == -1:
                    factors.append(r"-1")
                elif value != 1:
                    factors.append(str(value))

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
                power = c.get("power", 1)
                if power != 1:
                    factors.append(rf"{latex_6j(a,b,fval,cval,d,e)}^{{{power}}}")
                else:
                    factors.append(latex_6j(a, b, fval, cval, d, e))

            elif typ == "W6j":
                args = c.get("args")
                if not args:
                    args = (
                        fixed.get("a", fixed.get("A")),
                        fixed.get("b", fixed.get("B")),
                        fixed.get("e", fixed.get("E")),
                        fixed.get("c", fixed.get("C")),
                        fixed.get("d", fixed.get("D")),
                        fixed.get("f", fixed.get("F")),
                    )
                a, b, e, cval, d, fval = args
                power = c.get("power", 1)
                if power != 1:
                    factors.append(rf"{latex_6j(a,b,e,cval,d,fval)}^{{{power}}}_{{\mathrm{{W}}}}")
                else:
                    factors.append(rf"{latex_6j(a,b,e,cval,d,fval)}_{{\mathrm{{W}}}}")

            elif typ == "theta":
                args = c.get("args")
                if not args:
                    args = (
                        fixed.get("a", fixed.get("A")),
                        fixed.get("b", fixed.get("B")),
                        fixed.get("c", fixed.get("C"))
                    )
                a, b, cval = args
                power = c.get("power", 1)
                if power != 1:
                    factors.append(rf"\Theta\left({a},{b},{cval}\right)^{{{power}}}")
                else:
                    factors.append(rf"\Theta\left({a},{b},{cval}\right)")

            elif typ == "delta":
                j = fixed.get("j", fixed.get("J"))
                power = c.get("power", 1)
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


# Renders the expression as a single LaTeX string joined with \cdot and
# saves it to a PDF file using matplotlib. The figure is sized to fit the
# expression on one line (8×1 inches); increase figsize for longer formulas.
def save_latex_pdf(terms, filename="expression.pdf"):
    factors = latex_formatting(terms)
    flat_factors = [f for sublist in factors for f in sublist]
    expr_str = r" \cdot ".join(flat_factors)

    fig, ax = plt.subplots(figsize=(8, 1))
    ax.axis('off')
    ax.text(0.5, 0.5, f"${expr_str}$", ha='center', va='center', fontsize=16)
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved LaTeX expression to {filename}")


# -----------------------------------------------------------------------
# Plain-text formula serialisation (evaluate_formula.py compatible)
# -----------------------------------------------------------------------

def _sanitize_py(s: str) -> str:
    """Replace prime characters in sympy symbol names to make valid Python identifiers.

    e.g. "n'" -> "n_p", "n''" -> "n_pp", "x' - n" -> "x_p - n"
    """
    result = []
    i = 0
    while i < len(s):
        if s[i] == "'":
            # Replace run of primes with _p, _pp, etc.
            j = i
            while j < len(s) and s[j] == "'":
                j += 1
            result.append("_p" * (j - i))
            i = j
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _arg_str(val) -> str:
    """Convert a coefficient argument value to a sanitized Python string."""
    return _sanitize_py(str(val))


def _sign_coeff_to_exponent_str(sign_args):
    """
    Convert a list of (coeff_str, value) sign args to a Python exponent expression.

    coeff_str can be: "-", "+", None, or a numeric string like "2", "+2", "-3".
    Returns e.g. "a + 2*F_1 - b" ready for use in (-1)**round(...).
    """
    parts = []
    for coeff_str, val in sign_args:
        if coeff_str == '-':
            numeric_coeff = -1
        elif coeff_str in ('+', None):
            numeric_coeff = 1
        else:
            try:
                numeric_coeff = float(coeff_str)
            except (TypeError, ValueError):
                numeric_coeff = 1

        val_str = _arg_str(val)
        if numeric_coeff == 1:
            parts.append(f"+ {val_str}")
        elif numeric_coeff == -1:
            parts.append(f"- {val_str}")
        elif numeric_coeff > 0:
            parts.append(f"+ {numeric_coeff:g}*{val_str}")
        else:
            parts.append(f"- {abs(numeric_coeff):g}*{val_str}")

    if not parts:
        return "0"
    expr = " ".join(parts).lstrip("+ ").strip()
    return expr


def _coeff_depends_on_vars(c, var_names):
    """Return True if coefficient c references any of the named sum variables."""
    var_set = set(var_names)
    for a in c.get("args", ()):
        if isinstance(a, str) and a in var_set:
            return True
    fixed = c.get("fixed", {})
    for v in fixed.values():
        if isinstance(v, str) and v in var_set:
            return True
        if isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, str) and sub in var_set:
                            return True
                elif isinstance(item, str) and item in var_set:
                    return True
    return False


def _coeff_to_formula_str(c):
    """Convert a single non-sum coefficient dict to a Python expression string."""
    typ = c.get("type")

    if typ == "sign_value":
        val = c.get("value", 1)
        return str(int(val)) if float(val) == int(float(val)) else str(val)

    elif typ == "sign":
        sign_args = c.get("fixed", {}).get("args", [])
        exponent = _sign_coeff_to_exponent_str(sign_args)
        if exponent == "0":
            return "1"
        return f"(-1)**round({exponent})"

    elif typ == "theta":
        args = c.get("args", ())
        power = c.get("power", 1)
        a, b, cv = args
        a, b, cv = _arg_str(a), _arg_str(b), _arg_str(cv)
        if power == 1:
            return f"theta({a}, {b}, {cv})"
        return f"theta({a}, {b}, {cv}, {power})"

    elif typ == "delta":
        fixed = c.get("fixed", {})
        j = _arg_str(fixed.get("j", fixed.get("J")))
        power = c.get("power", 1)
        if power == 1:
            return f"delta({j})"
        return f"delta({j}, {power})"

    elif typ == "W6j":
        args = c.get("args", ())
        if not args:
            fixed = c.get("fixed", {})
            args = (
                fixed.get("a", fixed.get("A")),
                fixed.get("b", fixed.get("B")),
                fixed.get("e", fixed.get("E")),
                fixed.get("c", fixed.get("C")),
                fixed.get("d", fixed.get("D")),
                fixed.get("f", fixed.get("F")),
            )
        power = c.get("power", 1)
        args_str = ", ".join(_arg_str(a) for a in args)
        if power == 1:
            return f"W6j({args_str})"
        return f"W6j({args_str}, {power})"

    return "1"


def terms_to_formula_string(terms):
    """
    Convert canonical spin network terms to an evaluate_formula.py-compatible
    Python expression string.

    Handles sums (-> Sum('var', min, max, lambda var: ...)), signs, thetas,
    deltas, and W6j symbols. Multiple terms are joined with +.

    Returns a string such as:
        "(-1)**round(F_1) * theta(1, 2, F_1) * Sum('F_1', 0, 2, lambda F_1: delta(F_1) * W6j(1,2,F_1,3,4,5))"
    """
    term_strs = []

    for term in terms:
        coeffs = [c for c in term.get("coeffs", []) if isinstance(c, dict)]

        # Collect sum variables in declaration order: var -> (min_expr, max_expr)
        sum_vars = {}
        for c in coeffs:
            if c.get("type") == "sum":
                var = c.get("index")
                rng = c.get("range2", {})
                if rng.get("symbolic") and "symbolic_Fmin" in rng and "symbolic_Fmax" in rng:
                    fmin_expr = _sanitize_py(rng["symbolic_Fmin"])
                    fmax_expr = _sanitize_py(rng["symbolic_Fmax"])
                else:
                    fmin_v = rng.get("Fmin", 0) / 2
                    fmax_v = rng.get("Fmax", 0) / 2
                    fmin_expr = str(int(fmin_v)) if fmin_v == int(fmin_v) else str(fmin_v)
                    fmax_expr = str(int(fmax_v)) if fmax_v == int(fmax_v) else str(fmax_v)
                sum_vars[var] = (fmin_expr, fmax_expr)

        # Partition remaining coefficients: outer (constant) vs inner (sum-dependent)
        outer, inner = [], []
        for c in coeffs:
            if c.get("type") == "sum":
                continue
            if _coeff_depends_on_vars(c, sum_vars):
                inner.append(c)
            else:
                outer.append(c)

        # Build inner lambda body
        inner_parts = [_coeff_to_formula_str(c) for c in inner]
        inner_parts = [p for p in inner_parts if p != "1"]
        inner_str = " * ".join(inner_parts) if inner_parts else "1"

        # Wrap in Sum(...) — innermost variable last (reversed insertion order)
        sum_expr = inner_str
        for var, (fmin_s, fmax_s) in reversed(list(sum_vars.items())):
            sum_expr = f"Sum('{var}', {fmin_s}, {fmax_s}, lambda {var}: {sum_expr})"

        # Build full term
        outer_parts = [_coeff_to_formula_str(c) for c in outer]
        outer_parts = [p for p in outer_parts if p != "1"]
        all_parts = outer_parts + ([sum_expr] if sum_vars else [])
        term_str = " * ".join(all_parts) if all_parts else "1"
        term_strs.append(term_str)

    if not term_strs:
        return "0"
    if len(term_strs) == 1:
        return term_strs[0]
    return " + ".join(f"({s})" for s in term_strs)


def save_formula_txt(terms, filename):
    """
    Save canonical terms as an evaluate_formula.py-compatible text file.

    The file contains one Python expression on a single line (after comments)
    and can be used directly:
        python scripts/evaluate_formula.py "$(grep -v '^#' FILE)"
    or piped:
        python scripts/evaluate_formula.py --formula-file FILE
    """
    formula = terms_to_formula_string(terms)
    base = filename.replace("\\", "/").split("/")[-1]
    with open(filename, "w") as f:
        f.write(f"# Spin network expression — {base}\n")
        f.write(f"# Evaluate with: python scripts/evaluate_formula.py \"$(grep -v '^#' {base})\"\n")
        f.write(formula + "\n")
    print(f"Saved formula to {filename}")
    return formula
