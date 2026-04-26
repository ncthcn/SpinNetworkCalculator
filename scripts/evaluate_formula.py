"""
Standalone spin network formula evaluator.

Evaluate any expression involving theta, delta, W6j, and math functions.

USAGE:
    python scripts/evaluate_formula.py "theta(1, 1, 1)"
    python scripts/evaluate_formula.py "W6j(1, 1, 1, 1, 1, 1)"
    python scripts/evaluate_formula.py "theta(j, j, 0) * W6j(j,j,0,j,j,0)" --var j=1
    python scripts/evaluate_formula.py "Sum('F', 0, 2, lambda F: delta(F) * W6j(1, 2, F, 1, 2, 3))"

AVAILABLE FUNCTIONS:
    theta(j1, j2, j3)               Theta symbol = (-1)^(j1+j2+j3) * (j1+j2+j3+1)! / ...
    delta(j)                        Delta symbol = (-1)^(2j) * (2j+1)
    W6j(j1, j2, j3, j4, j5, j6)    Wigner 6j symbol  {j1 j2 j3 / j4 j5 j6}
    Sum('var', min, max, func)      Sum from min to max (step 1), e.g.:
                                      Sum('F', 0, 3, lambda F: delta(F))
    sqrt, exp, log, sin, cos, pi, abs, ... (standard math)

VARIABLE BINDING:
    --var NAME=VALUE                Bind a numeric variable in the formula.
                                    VALUE is parsed as a float (e.g. --var j=0.5)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spin_evaluator import FormulaEvaluator


def parse_var(s: str):
    """Parse 'name=value' into (name, float)."""
    if '=' not in s:
        raise argparse.ArgumentTypeError(f"Variable must be NAME=VALUE, got: {s!r}")
    name, _, value = s.partition('=')
    try:
        return name.strip(), float(value.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"Value must be a number, got: {value!r}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a spin network formula numerically.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "formula", nargs="?", default=None,
        help="Formula string to evaluate (omit if using --formula-file)"
    )
    parser.add_argument(
        "--formula-file", "-f", metavar="FILE",
        help="Read formula from a .txt file (lines starting with # are ignored)",
    )
    parser.add_argument(
        "--var", metavar="NAME=VALUE", action="append", default=[], type=parse_var,
        help="Bind a variable (repeatable). E.g. --var j=0.5 --var k=1"
    )
    parser.add_argument(
        "--max-j", type=int, default=100,
        help="Maximum j value for wigxjpf tables (default: 100)"
    )
    parser.add_argument(
        "--raw", action="store_true",
        help="Print raw float result (no rounding)"
    )

    args = parser.parse_args()

    # Resolve formula source
    file_vars = {}
    if args.formula_file:
        formula_lines = []
        with open(args.formula_file) as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("# var "):
                    # "# var name=value" declares a variable default
                    try:
                        decl = stripped[6:]
                        name, _, val = decl.partition("=")
                        file_vars[name.strip()] = float(val.strip())
                    except Exception:
                        pass
                elif not stripped.startswith("#"):
                    formula_lines.append(stripped)
        formula = " ".join(formula_lines)
    elif args.formula:
        formula = args.formula
    else:
        parser.error("Provide a formula string or --formula-file FILE")

    # CLI --var overrides file-declared variables
    variables = {**file_vars, **dict(args.var)}
    if variables:
        print(f"Variables: {variables}")

    if not formula.strip():
        if args.formula_file:
            print(f"Error: '{args.formula_file}' contains no formula (only comments or is empty).")
            print("Regenerate the file by re-running the script that produced it.")
        else:
            print("Error: empty formula string.")
        sys.exit(1)

    print(f"Formula:   {formula}\n")

    fe = FormulaEvaluator(max_two_j=args.max_j * 2)
    try:
        result = fe.evaluate(formula, variables=variables or None)
    finally:
        fe.cleanup()

    if args.raw:
        print(f"Result: {result}")
    else:
        # Show both exact float and rounded integer if close to one
        rounded = round(result)
        if abs(result - rounded) < 1e-6:
            print(f"Result: {result:.10g}  (≈ {rounded})")
        else:
            print(f"Result: {result:.10g}")


if __name__ == "__main__":
    main()
