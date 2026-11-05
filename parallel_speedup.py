"""Strong-scaling test of differential evolution on the Earth-Mars problem.

The same search (same seed, same number of generations) is run with an increasing
number of worker processes. For each run we report the median wall-clock time,
the speedup S(p) = T(1) / T(p) and the efficiency E(p) = S(p) / p.

The cost function of trajectory_optimisation.py is very cheap (a fraction of a
millisecond), so the cost of distributing the work would hide any gain. To
mimic a more expensive trajectory model we evaluate it `reps` times per call:
this is an artificial load, and it is declared as such in the README.
"""
import os
import time
from functools import partial

import numpy as np
from scipy.optimize import differential_evolution

from trajectory_optimisation import A_EARTH, A_MARS, cost


def heavy_cost(x, reps):
    """The Earth-Mars cost evaluated `reps` times (artificial load)."""
    c = 0.0
    for _ in range(reps):
        c = cost(x)
    return c


REPS = 20          # artificial load: cost evaluations per call
GENERATIONS = 20  # same amount of work in every run
POPSIZE = 40      # individuals = POPSIZE x number of parameters
REPEATS = 3       # repetitions per setting (the median is reported)


def run(workers, reps, generations=GENERATIONS, popsize=POPSIZE):
    t_syn = 1 / abs(1 - (A_EARTH / A_MARS) ** 1.5)
    start = time.perf_counter()
    res = differential_evolution(
        partial(heavy_cost, reps=reps), [(0.0, t_syn), (0.3, 1.2)],
        seed=1, popsize=popsize, maxiter=generations, tol=0.0, polish=False,
        updating="deferred", workers=workers)
    return time.perf_counter() - start, res.fun


def measure(worker_list, reps, repeats=REPEATS):
    times = {}
    for w in worker_list:
        runs = [run(w, reps)[0] for _ in range(repeats)]
        times[w] = float(np.median(runs))
    return times


if __name__ == "__main__":
    n_cores = os.cpu_count()
    worker_list = [1] + [w for w in (2, 4, 8, 16) if w <= n_cores]
    reps = REPS
    print(f"cores available: {n_cores}, load: {reps} evaluations per call")

    times = measure(worker_list, reps)
    t1 = times[1]
    print(f"{'workers':>8} {'time (s)':>10} {'speedup':>9} {'efficiency':>11}")
    for w in worker_list:
        s = t1 / times[w]
        print(f"{w:>8} {times[w]:>10.2f} {s:>9.2f} {s / w:>11.2f}")

    # parallel fraction implied by the largest run (Amdahl: S = 1 / ((1-f) + f/p))
    p = worker_list[-1]
    if p > 1:
        s = t1 / times[p]
        f = (1 - 1 / s) / (1 - 1 / p)
        print(f"parallel fraction implied by Amdahl at p = {p}: f = {f:.2f}")

    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    s_list = [t1 / times[w] for w in worker_list]
    plt.figure(figsize=(6, 4.5))
    plt.plot(worker_list, worker_list, "k--", label="ideal")
    plt.plot(worker_list, s_list, "o-", label="measured")
    plt.xlabel("worker processes")
    plt.ylabel("speedup")
    plt.title("Differential evolution: strong scaling")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("figures/speedup.png", dpi=150, bbox_inches="tight")
