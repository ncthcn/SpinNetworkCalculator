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
Measure where parallel evaluation actually starts to pay off.

Splitting a summation across processes costs a fixed amount (spawning the
workers, and re-allocating the wigxjpf tables in each one) and saves a
fraction of the summation work. Below some number of iterations the fixed cost
dominates and the "parallel" backend is simply slower.

This script measures both sides of that trade so the threshold in
FormulaEvaluator._MIN_ITERATIONS_TO_PARALLELISE is an empirical number rather
than a guess, and prints the resulting break-even point.

    python scripts/benchmark_backends.py
"""

import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multiprocessing import Pool, cpu_count

from src.spin_evaluator import (
    FormulaEvaluator,
    _parallel_worker_init,
    _parallel_worker_evaluate_variables,
)

MAX_TWO_J = 200
REPEATS = 5


def formula(n_iterations):
    """One summation of `n_iterations` terms, each with a 6j and a theta."""
    return (
        f"Sum('F_1', 0.0, {float(n_iterations - 1)}, lambda F_1: "
        f"delta(F_1) * W6j(F_1,10,10,10,10,10) * theta(F_1,10,10))"
    )


def best_of(fn, repeats=REPEATS):
    """Minimum wall time over `repeats` runs -- least sensitive to noise."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return min(timings)


def measure_pool_startup(n_workers):
    """Cost of standing up a worker pool that has initialised wigxjpf."""
    def run():
        with Pool(
            processes=n_workers,
            initializer=_parallel_worker_init,
            initargs=(MAX_TWO_J,),
        ) as pool:
            # One trivial task per worker, so every worker really initialises.
            pool.map(_parallel_worker_evaluate_variables,
                     [("1.0", None)] * n_workers)
    return best_of(run, repeats=3)


def main():
    n_workers = max(1, cpu_count() - 1)
    print(f"machine       : {platform.platform()}")
    print(f"python        : {platform.python_version()}")
    print(f"cpu_count     : {cpu_count()}  (workers used: {n_workers})")
    print(f"max_two_j     : {MAX_TWO_J}")
    print(f"timing        : best of {REPEATS} runs")
    print()

    serial = FormulaEvaluator(max_two_j=MAX_TWO_J, backend="serial", verbose=False)
    try:
        print("--- serial cost per summation term ---")
        per_iteration = None
        for n in (200, 2_000, 20_000):
            elapsed = best_of(lambda: serial.evaluate(formula(n)),
                              repeats=3 if n >= 20_000 else REPEATS)
            per_iteration = elapsed / n
            print(f"  {n:>7,} iterations : {elapsed * 1000:9.2f} ms "
                  f"({per_iteration * 1e6:6.2f} us/iteration)")
    finally:
        serial.cleanup()

    print()
    print("--- fixed cost of going parallel ---")
    startup = measure_pool_startup(n_workers)
    print(f"  pool startup + wigxjpf init in {n_workers} workers: "
          f"{startup * 1000:.1f} ms")

    print()
    print("--- break-even ---")
    # Parallel wins when   startup + T_serial/n_workers  <  T_serial
    # i.e. when  T_serial  >  startup * n_workers / (n_workers - 1)
    if n_workers <= 1:
        print("  only one worker available; parallelism cannot help")
        return

    break_even_seconds = startup * n_workers / (n_workers - 1)
    break_even_iterations = break_even_seconds / per_iteration
    print(f"  parallel only wins above ~{break_even_seconds * 1000:.0f} ms "
          f"of serial work")
    print(f"  which is ~{break_even_iterations:,.0f} summation terms")
    print()
    print(f"  configured _PARALLEL_STARTUP_SECONDS = "
          f"{FormulaEvaluator._PARALLEL_STARTUP_SECONDS} s "
          f"(measured here: {startup:.2f} s)")

    print()
    print("--- end-to-end check at a size where parallelism should win ---")
    big_n = int(break_even_iterations * 8)
    print(f"  using {big_n:,} summation terms")

    serial = FormulaEvaluator(max_two_j=MAX_TWO_J, backend="serial", verbose=False)
    parallel = FormulaEvaluator(max_two_j=MAX_TWO_J, backend="multiprocessing",
                                verbose=False)
    try:
        expr = formula(big_n)
        t0 = time.perf_counter()
        v_serial = serial.evaluate(expr)
        t_serial = time.perf_counter() - t0

        t0 = time.perf_counter()
        v_parallel = parallel.evaluate(expr)
        t_parallel = time.perf_counter() - t0

        print(f"  serial          : {t_serial:6.2f} s   -> {v_serial!r}")
        print(f"  multiprocessing : {t_parallel:6.2f} s   -> {v_parallel!r}")
        print(f"  identical       : {v_serial == v_parallel}")
        print(f"  speedup         : {t_serial / t_parallel:.2f}x")
    finally:
        serial.cleanup()
        parallel.cleanup()


if __name__ == "__main__":
    main()
