"""Sampled vs linear statistical volume for the Kepler problem.

Compares sqrt(det P) of the sampled covariance of a propagated Gaussian cloud
with the linear value sqrt(det P0) (constant, since det Phi = 1).
Two propagators: DOP853 (adaptive, parallel over samples) and Stormer-Verlet
(fixed step, vectorised over samples).

Usage:
    python3 sampled_vs_linear_volume.py              # tables of ratios
    python3 sampled_vs_linear_volume.py --speedup [N]  # strong-scaling timing
"""
import os
import sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from multiprocessing import Pool

n, mu = 2, 1.0   # spatial dimension, gravitational parameter


# ---------- dynamics ----------
def rhs_state(t, y):
    """Right-hand side for a single state y = (q, v), used by solve_ivp."""
    q, v = y[:n], y[n:]
    r = np.linalg.norm(q)
    return np.concatenate([v, -mu * q / r**3])


def initial_state(e):
    """Periapsis state, a = 1: q0 = (1-e, 0), v0 = (0, sqrt((1+e)/(1-e)))."""
    q0 = (1 - e, 0)
    v0 = (0, np.sqrt((1 + e) / (1 - e)))
    return np.concatenate([q0, v0])


def acceleration(q):
    """Gravitational acceleration for all samples. q: (N, 2) -> (N, 2)."""
    r = np.linalg.norm(q, axis=1, keepdims=True)
    return -mu * q / r**3


# ---------- DOP853, parallel over samples ----------
def propagate_one_sample(args):
    x0, t_list, tol = args
    sol = solve_ivp(rhs_state, (0, t_list[-1]), x0, method="DOP853",
                    t_eval=t_list, rtol=tol, atol=tol)
    return sol.y.T                      # (len(t_list), 2n)


def propagate_cloud(X0, t_list, tol=1e-12, n_proc=4):
    """X0: (N, 2n). Returns array (len(t_list), N, 2n)."""
    tasks = [(x0, t_list, tol) for x0 in X0]
    with Pool(processes=n_proc) as pool:
        results = pool.map(propagate_one_sample, tasks, chunksize=50)
    return np.stack(results, axis=1)


# ---------- Stormer-Verlet, vectorised over samples ----------
def verlet_step(q, v, h):
    v_half = v + 0.5 * h * acceleration(q)
    q_new = q + h * v_half
    v_new = v_half + 0.5 * h * acceleration(q_new)
    return q_new, v_new


def propagate_verlet(X0, periods, steps_per_period):
    """X0: (N, 4). Returns array (periods, N, 4), one state per full period."""
    q, v = X0[:, :2].copy(), X0[:, 2:].copy()
    h = 2 * np.pi / steps_per_period
    out = []
    for _ in range(periods):
        for _ in range(steps_per_period):
            q, v = verlet_step(q, v, h)
        out.append(np.hstack([q, v]))
    return np.array(out)


# ---------- statistics and driver ----------
def sqrt_det_cov(X):
    """sqrt(det) of the sample covariance of X, shape (N, 2n)."""
    return np.sqrt(np.linalg.det(np.cov(X.T)))

def cartesian_to_delaunay(X, mu=1.0):
    """Planar Delaunay variables (l, g, L, G) from Cartesian states (q, v).

    X: (N, 4). Returns (N, 4) with columns (l, g, L, G): mean anomaly, argument
    of periapsis, L = sqrt(mu a), G = angular momentum. Elliptic orbits, e > 0.
    """
    q, v = X[:, :2], X[:, 2:]
    r = np.linalg.norm(q, axis=1)
    v2 = np.sum(v**2, axis=1)
    a = 1.0 / (2.0 / r - v2 / mu)
    L = np.sqrt(mu * a)
    G = q[:, 0] * v[:, 1] - q[:, 1] * v[:, 0]
    e = np.sqrt(np.maximum(1.0 - (G / L)**2, 0.0))
    e_x = ((v2 - mu/r) * q[:,0] - (q[:,0] * v[:,0] + q[:,1] * v[:,1]) * v[:,0]) / mu
    e_y = ((v2 - mu/r) * q[:,1] - (q[:,0] * v[:,0] + q[:,1] * v[:,1])* v[:,1]) / mu
    g =  np.arctan2(e_y, e_x)
    theta = np.arctan2(q[:,1], q[:,0])
    nu = np.angle(np.exp(1j*(theta - g)))
    E = np.arctan2(np.sqrt(1 - e**2) * np.sin(nu), e + np.cos(nu))
    l = E - e * np.sin(E)
    return np.column_stack([l, g, L, G])

def wrap(x):
    return np.angle(np.exp(1j * x))

def sqrt_det_cov_delaunay(D):
    """D: (N, 4) with columns (l, g, L, G). sqrt(det) of the covariance,
    with circular centring of the two angles."""
    D = D.copy()
    for k in (0, 1):                                   # angle columns: l, g
        mean = np.angle(np.mean(np.exp(1j * D[:, k])))  # circular mean
        D[:, k] = wrap(D[:, k] - mean)                  # centre, then wrap
    return np.sqrt(np.linalg.det(np.cov(D.T)))

def run(sigma, N, periods, e=0.7, n_proc=4, seed=0, t_list=None,
        method="dop853", steps_per_period=400, coords="cartesian",
        tol=1e-12, reference="theory"):
    """Ratio sampled / linear volume at each requested time.

    coords:    "cartesian" or "delaunay".
    reference: "theory" -> sqrt(det P0) (Cartesian only); "sample" -> the same
               samples at t = 0, which removes the sampling noise of the
               covariance. Delaunay always uses the sample reference.
    tol:       rtol = atol of DOP853.
    """
    rng = np.random.default_rng(seed)
    x0 = initial_state(e)
    P0 = np.diag([sigma**2] * (2 * n))
    X0 = rng.multivariate_normal(x0, P0, size=N)
    if t_list is None:
        t_list = np.arange(1, periods + 1) * 2 * np.pi
    if method == "dop853":
        traj = propagate_cloud(X0, t_list, tol=tol, n_proc=n_proc)
    elif method == "verlet":
        traj = propagate_verlet(X0, periods, steps_per_period)
    else:
        raise ValueError(f"unknown method: {method}")
    if coords == "cartesian":
        if reference == "theory":
            ref = np.sqrt(np.linalg.det(P0))
        elif reference == "sample":
            ref = sqrt_det_cov(X0)
        else:
            raise ValueError(f"unknown reference: {reference}")
        return [sqrt_det_cov(traj[j]) / ref for j in range(len(t_list))]
    elif coords == "delaunay":
        ref = sqrt_det_cov_delaunay(cartesian_to_delaunay(X0))   # same samples at t = 0
        return [sqrt_det_cov_delaunay(cartesian_to_delaunay(traj[j])) / ref
                for j in range(len(t_list))]
    else:
        raise ValueError(f"unknown coords: {coords}")


def _best_time(f, repeats=3):
    """Smallest wall-clock time of f() over a few repetitions."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t0)
    return best


def speedup(N=4000, sigma=1e-8, e=0.7, tol=1e-12, procs=(1, 2, 4, 8)):
    """Strong scaling of the DOP853 propagation over one period.

    Times propagate_cloud for a fixed number of samples against a serial
    loop without multiprocessing, and compares with vectorised Verlet.
    Run it on the machine to be measured."""
    rng = np.random.default_rng(0)
    X0 = rng.multivariate_normal(initial_state(e), np.diag([sigma**2] * (2 * n)), size=N)
    t_list = np.array([2 * np.pi])
    print(f"N = {N} samples, 1 period, logical CPUs: {os.cpu_count()}")
    T_serial = _best_time(lambda: [propagate_one_sample((x, t_list, tol)) for x in X0])
    print(f"serial loop (no Pool): {T_serial:.3f} s")
    print(f"{'processes':>9} {'time [s]':>9} {'speedup':>8} {'efficiency':>10}")
    for p in procs:
        Tp = _best_time(lambda: propagate_cloud(X0, t_list, tol=tol, n_proc=p))
        print(f"{p:>9} {Tp:9.3f} {T_serial / Tp:8.2f} {T_serial / Tp / p:10.2f}")
    T_v = _best_time(lambda: propagate_verlet(X0, 1, 1600))
    print(f"Stormer-Verlet, 1600 steps/period, vectorised: {T_v:.3f} s "
          f"({T_serial / T_v:.1f}x faster than the serial DOP853 loop)")


if __name__ == "__main__":
    if "--speedup" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--speedup"]
        speedup(int(args[0]) if args else 4000)
        sys.exit()
    for sigma in [1e-10, 1e-9, 1e-8, 1e-6]:
        for coords in ["cartesian", "delaunay"]:
            ref = run(sigma, N=2000, periods=1, coords=coords)
            print("DOP853", coords, sigma, [float(r) for r in ref])
            for m in [400, 800, 1600, 3200]:
                res = run(sigma, N=2000, periods=1, method="verlet",
                          steps_per_period=m, coords=coords)
                print("Verlet", m, coords, sigma, [float(r) for r in res])
