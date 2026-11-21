"""Arenstorf orbit of the planar circular restricted three-body problem.

Integrates one period with DOP853 (rtol = atol = 1e-12) and saves two figures:
- figures/pcr3bp_arenstorf.png: the orbit in the rotating frame with the two primaries,
  and the variation of the Jacobi constant along it;
- figures/pcr3bp_jacobi_curves.png: zero-velocity curves at the values of C of L1, L2, L3
  with the orbit, and the forbidden region C(x, y, v = 0) < C for an example value of C
  (as in three_body_problem.m of the MATLAB repository).

Usage:
    python3 pcr3bp_orbit.py [--outdir figures]
"""
import argparse
import os

import numpy as np
from scipy.integrate import solve_ivp

from pcr3bp import (MU, T_ARENSTORF, Y0_ARENSTORF, jacobi, lagrange_points, rhs,
                    zero_velocity_jacobi)


def arenstorf_orbit(n=4000):
    """Times and states (n + 1, 4) of the Arenstorf orbit over one period."""
    t = np.linspace(0.0, T_ARENSTORF, n + 1)
    sol = solve_ivp(rhs, (0.0, T_ARENSTORF), Y0_ARENSTORF, method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t)
    return sol.t, sol.y.T


def save_jacobi_curves(outdir, c_example=3.10, h=0.005):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, Y = arenstorf_orbit()
    c_orbit = jacobi(Y[0])
    L = lagrange_points()
    c_levels = {f"L{k + 1}": zero_velocity_jacobi(*L[k]) for k in (0, 1, 2, 3)}
    g = np.arange(-1.5, 1.5 + h / 2, h)
    X, Yg = np.meshgrid(g, g)
    Cgrid = zero_velocity_jacobi(X, Yg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    # left: the orbit and the zero-velocity curves at the critical values of C
    colors = ["tab:red", "tab:orange", "tab:green"]
    for color, (name, c) in zip(colors, list(c_levels.items())[:3]):
        ax1.contour(X, Yg, Cgrid, levels=[c], colors=color, linewidths=1.0)
        ax1.plot([], [], color=color, label=f"zero-velocity curve at C({name}) = {c:.4f}")
    ax1.plot(Y[:, 0], Y[:, 1], "b", lw=1.0, label=f"Arenstorf orbit, C = {c_orbit:.4f}")
    ax1.set_title(f"C of the orbit < C(L4) = {c_levels['L4']:.4f}: nothing is forbidden")
    # right: forbidden region for an example value of C
    ax2.contourf(X, Yg, Cgrid, levels=[0, c_example], colors=["0.85"])
    ax2.contour(X, Yg, Cgrid, levels=[c_example], colors="k", linewidths=0.8)
    ax2.plot([], [], "s", color="0.85", mec="k", label="forbidden region")
    ax2.set_title(f"Forbidden region for C = {c_example:.2f}")
    for ax in (ax1, ax2):
        ax.plot(L[:, 0], L[:, 1], "kx", ms=8, label="Lagrange points")
        ax.plot(-MU, 0, "o", color="tab:red", ms=8, label="Earth")
        ax.plot(1 - MU, 0, "o", color="tab:blue", ms=5, label="Moon")
        ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xlabel="x", ylabel="y")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "pcr3bp_jacobi_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved in {path}")
    for name, c in c_levels.items():
        print(f"  C({name}) = {c:.6f}")


def save_figure(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, Y = arenstorf_orbit()
    C = jacobi(Y)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(Y[:, 0], Y[:, 1], lw=1.2, label="Arenstorf orbit")
    ax1.plot(-MU, 0, "o", color="tab:blue", ms=9, label="Earth")
    ax1.plot(1 - MU, 0, "o", color="gray", ms=5, label="Moon")
    ax1.plot(Y[0, 0], Y[0, 1], "g.", ms=12, label="start")
    ax1.set(title="Rotating frame, one period", xlabel="x", ylabel="y", aspect="equal")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")
    ax2.plot(t, C - C[0])
    ax2.set(title=f"Jacobi constant, C(0) = {C[0]:.4f}", xlabel="t", ylabel="C(t) - C(0)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "pcr3bp_arenstorf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Figure saved in {path}")
    print(f"Max |C(t) - C(0)| = {np.max(np.abs(C - C[0])):.2e}, "
          f"distance from the initial state at t = T: {np.linalg.norm(Y[-1] - Y0_ARENSTORF):.2e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--outdir", default="figures")
    outdir = parser.parse_args(argv).outdir
    save_figure(outdir)
    save_jacobi_curves(outdir)


if __name__ == "__main__":
    main()