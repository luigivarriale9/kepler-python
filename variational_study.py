"""Symplecticity of the state-transition matrix and validity of linear covariance
propagation for the Kepler problem (e = 0.7).  Saves figures/variational_symplecticity.png
and figures/variational_covariance.png."""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from kepler import initial_state
from variational import (METHODS_AUG, propagate, symplecticity_error, unpack, pack, rhs,
                         jacobian, initial_augmented)

E_ECC, PERIOD = 0.7, 2 * np.pi
X0 = initial_state(E_ECC)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def symplecticity_study(n_periods=20, steps_per_period=500):
    """|| Phi^T J Phi - J || and det Phi - 1 at the end of every period."""
    n = n_periods * steps_per_period
    res = {}
    for name, step in METHODS_AUG.items():
        _, W = propagate(step, X0, n_periods * PERIOD, n)
        Ph = [unpack(w)[1] for w in W[::steps_per_period]]
        res[name] = (np.array([symplecticity_error(p) for p in Ph]),
                     np.array([np.linalg.det(p) - 1 for p in Ph]))
    return res


def covariance_study(sigmas=(1e-5, 1e-4, 1e-3), n_periods=10, n_samples=2000, seed=0):
    """Relative Frobenius error between the sample covariance of a Monte Carlo cloud (exact
    flow, DOP853) and the linear propagation Phi P0 Phi^T, with P0 = sigma^2 I."""
    times = PERIOD * np.arange(1, n_periods + 1)
    f = lambda t, w: pack(rhs(t, unpack(w)[0]), jacobian(unpack(w)[0]) @ unpack(w)[1])
    sol = solve_ivp(f, (0, times[-1]), initial_augmented(X0), method="DOP853",
                    rtol=1e-13, atol=1e-14, t_eval=times)
    Phis = [sol.y[4:, k].reshape(4, 4) for k in range(len(times))]

    def batch(t, y):
        X = y.reshape(4, n_samples)
        r = np.linalg.norm(X[:2], axis=0)
        return np.vstack([X[2:], -X[:2] / r**3]).ravel()

    Z = np.random.default_rng(seed).standard_normal((4, n_samples))
    out = {}
    for sig in sigmas:
        X0s = X0[:, None] + sig * Z
        s = solve_ivp(batch, (0, times[-1]), X0s.ravel(), method="DOP853",
                      rtol=1e-12, atol=1e-13, t_eval=times)
        err = []
        for k, Ph in enumerate(Phis):
            X = s.y[:, k].reshape(4, n_samples)
            D = X - X.mean(axis=1, keepdims=True)
            Pmc, Plin = D @ D.T / (n_samples - 1), sig**2 * Ph @ Ph.T
            err.append(np.linalg.norm(Pmc - Plin) / np.linalg.norm(Plin))
        out[sig] = np.array(err)
    return times / PERIOD, out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = symplecticity_study()
    print("Symplecticity error ||Phi^T J Phi - J|| after 1, 5, 20 periods (500 steps/period)")
    for name, (s, d) in res.items():
        print(f"  {name:18s} {s[1]:9.2e} {s[5]:9.2e} {s[20]:9.2e}   det Phi - 1 at 20: {d[20]:9.2e}")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, (s, _) in res.items():
        ax.semilogy(np.arange(len(s)), np.maximum(s, 1e-17), marker="o", ms=3, label=name)
    ax.set_xlabel("periods"); ax.set_ylabel(r"$\|\Phi^T J \Phi - J\|_F$")
    ax.set_title("Symplecticity of the state-transition matrix (e = 0.7)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "variational_symplecticity.png"), dpi=150)

    t, cov = covariance_study()
    print("\nRelative error of the linear covariance vs Monte Carlo (N = 2000), exact flow")
    for sig, e in cov.items():
        print(f"  sigma = {sig:.0e}: " + " ".join(f"{e[k]:.2f}" for k in (0, 1, 4, 9)) + "  (after 1, 2, 5, 10 periods)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sig, e in cov.items():
        ax.semilogy(t, e, marker="o", ms=3, label=rf"$\sigma$ = {sig:.0e}")
    ax.axhline(0.1, color="k", ls=":", lw=1)
    ax.set_xlabel("periods"); ax.set_ylabel("relative covariance error")
    ax.set_title("Linear covariance propagation vs Monte Carlo (e = 0.7)")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "variational_covariance.png"), dpi=150)
