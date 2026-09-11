#!/usr/bin/env python
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
Verify that the multiprocessing backend produces exactly the same numbers as
the serial one, and report the speedup.

Run as a real script (not from a heredoc or a notebook): the 'spawn' start
method used on macOS re-imports __main__ in every worker, so the entry point
must be an importable file guarded by `if __name__ == "__main__"`.

    python scripts/check_backends.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spin_evaluator import FormulaEvaluator, _multiprocessing_is_usable

# A single outer summation of 200 terms, each involving a 6j and a theta.
# 200 is well above the 64-iteration threshold at which parallelising pays off.
BIG_SUM = (
    "Sum('F_1', 0.0, 199.0, lambda F_1: "
    "delta(F_1) * W6j(F_1,10,10,10,10,10) * theta(F_1,10,10))"
)

# The outer sum is squared, so chunking it would be invalid; the evaluator must
# detect this and stay serial.
NON_LINEAR = "Sum('F_1', 0.0, 199.0, lambda F_1: delta(F_1)) ** 2"


def timed(evaluator, formula):
    start = time.perf_counter()
    value = evaluator.evaluate(formula)
    return value, time.perf_counter() - start


def main():
    print(f"multiprocessing usable here: {_multiprocessing_is_usable()}")
    print(f"start method __main__ file : {getattr(sys.modules['__main__'], '__file__', None)}")
    print()

    serial = FormulaEvaluator(max_two_j=200, backend="serial", verbose=False)
    parallel = FormulaEvaluator(max_two_j=200, backend="multiprocessing", verbose=False)
    try:
        print(f"linear-factor check : {parallel._outer_sum_is_a_linear_factor(BIG_SUM)}")
        print(f"probed outer bounds : {parallel._probe_outer_sum(BIG_SUM, None)}")
        print()

        v_serial, t_serial = timed(serial, BIG_SUM)
        v_par, t_par = timed(parallel, BIG_SUM)

        print(f"serial          = {v_serial!r}   ({t_serial * 1000:7.1f} ms)")
        print(f"multiprocessing = {v_par!r}   ({t_par * 1000:7.1f} ms)")
        print(f"bitwise identical: {v_serial == v_par}")
        if t_par > 0:
            print(f"speedup         : {t_serial / t_par:.2f}x")
        print()

        print(f"non-linear formula chunkable? "
              f"{parallel._outer_sum_is_a_linear_factor(NON_LINEAR)}  (must be False)")
        nl_serial = serial.evaluate(NON_LINEAR)
        nl_par = parallel.evaluate(NON_LINEAR)
        print(f"non-linear agrees: {nl_serial == nl_par}  ({nl_serial!r})")
    finally:
        serial.cleanup()
        parallel.cleanup()


if __name__ == "__main__":
    main()
