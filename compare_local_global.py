"""Local versus global search on the Earth-Mars Lambert cost.

Local method: scipy.optimize.minimize (L-BFGS-B, gradient by finite differences),
started from many random points. Global method: differential evolution.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from trajectory_optimisation import cost, hohmann_cost, A_EARTH, A_MARS, AU_PER_YR_KMS

T_SYN = 1 / abs(1 - (A_EARTH / A_MARS) ** 1.5)
BOUNDS = [(0.0, T_SYN), (0.3, 1.2)]
TARGET = hohmann_cost() * AU_PER_YR_KMS   # km/s


class Counter:
    def __init__(self):
        self.n = 0

    def __call__(self, x):
        self.n += 1
        return cost(x)


def local_runs(n_starts=100, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in BOUNDS]); hi = np.array([b[1] for b in BOUNDS])
    results = []
    for x0 in lo + rng.random((n_starts, 2)) * (hi - lo):
        f = Counter()
        res = minimize(f, x0, method="L-BFGS-B", bounds=BOUNDS)
        results.append((res.fun, f.n))
    return np.array(results)


def global_run(seed=1):
    f = Counter()
    res = differential_evolution(f, BOUNDS, seed=seed, tol=1e-10, popsize=20,
                                 maxiter=300, polish=False)
    return res.fun * AU_PER_YR_KMS, f.n


if __name__ == "__main__":
    r = local_runs()
    r[:, 0] *= AU_PER_YR_KMS
    ok = np.abs(r[:, 0] - TARGET) < 1e-2
    print(f"Hohmann reference: {TARGET:.3f} km/s")
    print(f"Local search (L-BFGS-B, {len(r)} random starts)")
    print(f"  reached the global minimum: {ok.sum()} of {len(r)}")
    print(f"  best {r[:,0].min():.3f}, worst {r[:,0].max():.3f} km/s")
    print(f"  mean evaluations per run: {r[:,1].mean():.0f}")
    gf, gn = global_run()
    print(f"Differential evolution: {gf:.3f} km/s, {gn} evaluations")
    for seed in (2, 3, 4):
        f, n = global_run(seed)
        print(f"  seed {seed}: {f:.3f} km/s, {n} evaluations")
