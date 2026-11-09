"""Sampled vs linear statistical volume for the Kepler problem.

Compares sqrt(det P) of the sampled covariance of a propagated Gaussian cloud
with the linear value sqrt(det P0) (constant, since det Phi = 1).
Two propagators: DOP853 (adaptive, parallel over samples) and Stormer-Verlet
(fixed step, vectorised over samples).
"""
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


def run(sigma, N, periods, e=0.7, n_proc=4, seed=0, t_list=None,
        method="dop853", steps_per_period=400):
    """Ratio sampled / linear volume at each requested time."""
    rng = np.random.default_rng(seed)
    x0 = initial_state(e)
    P0 = np.diag([sigma**2] * (2 * n))
    X0 = rng.multivariate_normal(x0, P0, size=N)
    if t_list is None:
        t_list = np.arange(1, periods + 1) * 2 * np.pi
    if method == "dop853":
        traj = propagate_cloud(X0, t_list, n_proc=n_proc)
    elif method == "verlet":
        traj = propagate_verlet(X0, periods, steps_per_period)
    else:
        raise ValueError(f"unknown method: {method}")
    vol_lin = np.sqrt(np.linalg.det(P0))
    return [sqrt_det_cov(traj[j]) / vol_lin for j in range(len(t_list))]


if __name__ == "__main__":
    for sigma in [1e-10, 1e-9, 1e-8, 1e-6]:
        ref = run(sigma, N=2000, periods=1)                       # DOP853
        print("DOP853", sigma, [float(r) for r in ref])
        for m in [400, 800, 1600, 3200]:
            res = run(sigma, N=2000, periods=1, method="verlet",
                      steps_per_period=m)
            print("Verlet", m, sigma, [float(r) for r in res])
