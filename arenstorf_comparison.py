"""Comparison of integrators on the Arenstorf orbit (planar circular restricted three-body problem).

Transposed from symplectic_integrators/arenstorf_orbit.m of the MATLAB repository.  The
model is pcr3bp.py (rotating frame, normalised units, Earth-Moon mass ratio).  The state
is (x, y, vx, vy); for the partitioned methods q = (x, y) and p = (vx, vy), so the Coriolis
term makes the acceleration depend on p and Stormer-Verlet is only first order here.

Methods: ODE45-like reference (RK45, rtol = atol = 1e-8), implicit midpoint, explicit
Euler, Stormer-Verlet (p explicit) and symplectic Euler (p implicit).  The energy error is
measured against the reference, as in the MATLAB program.

Usage:
    python3 arenstorf_comparison.py [--n 300000] [--methods NAME ...] [--no-plots]
"""
import argparse
import os
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

from kepler import euler_step, integrate, midpoint_step_fsolve, symplectic_euler_step, verlet_step
from pcr3bp import MU, T_ARENSTORF, Y0_ARENSTORF, acceleration, rhs

XTOL = 1e-12        # tolerance of the implicit solves, as in the MATLAB code (TolX)
REFERENCE = "ODE45"
NAMES = [REFERENCE, "Implicit midpoint", "Explicit Euler", "Stormer-Verlet", "Symplectic Euler"]

def energy(Y):
    """Hamiltonian of the rotating frame written with the velocities. Y: (N, 4) -> (N,)."""
    x, y, vx, vy = Y.T
    r1 = np.hypot(x + MU, y)
    r2 = np.hypot(x - 1 + MU, y)
    return 0.5 * (vx**2 + vy**2) - 0.5 * (x**2 + y**2) - (1 - MU) / r1 - MU / r2


def solve(name, n, t_end=T_ARENSTORF, y0=Y0_ARENSTORF):
    """Solution of the given method on n + 1 equally spaced times in [0, t_end]; shape (n+1, 4)."""
    if name == REFERENCE:
        tt = np.linspace(0.0, t_end, n + 1)
        return solve_ivp(rhs, (0.0, t_end), y0, method="RK45", rtol=1e-8, atol=1e-8, t_eval=tt).y.T
    steps = {
        "Implicit midpoint": partial(midpoint_step_fsolve, f=rhs, xtol=XTOL),
        "Explicit Euler": partial(euler_step, f=rhs),
        "Stormer-Verlet": partial(verlet_step, acc=acceleration),
        "Symplectic Euler": partial(symplectic_euler_step, acc=acceleration, implicit=True),
    }
    return integrate(steps[name], y0, t_end, n)


def compare(n, names):
    """Return {name: Y} for the requested methods (the reference is always included)."""
    names = [REFERENCE] + [m for m in names if m != REFERENCE]
    return {name: solve(name, n) for name in names}


def print_report(solutions, n):
    h = T_ARENSTORF / n
    E = {name: energy(Y) for name, Y in solutions.items()}
    print(f"Maximum energy error with respect to {REFERENCE} (h = {h:.3e}):")
    for name in solutions:
        if name != REFERENCE:
            print(f"  {name:18s} {np.max(np.abs(E[name] - E[REFERENCE])):.3e}")
    print("Distance from y0 after one period:")
    for name, Y in solutions.items():
        print(f"  {name:18s} {np.linalg.norm(Y[-1] - Y0_ARENSTORF):.3e}")


def save_figures(solutions, n, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, Y) in zip(axes.ravel(), solutions.items()):
        ax.plot(Y[:, 0], Y[:, 1], lw=1.5)
        ax.plot(Y[0, 0], Y[0, 1], "g.", ms=12)
        ax.plot(Y[-1, 0], Y[-1, 1], "r.", ms=12)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(solutions):]:
        ax.axis("off")
    fig.suptitle(f"Arenstorf orbit, one period, n = {n} steps")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "arenstorf_orbits.png"), dpi=150)
    plt.close(fig)

    tt = np.linspace(0.0, T_ARENSTORF, n + 1)
    E = {name: energy(Y) for name, Y in solutions.items()}
    e0 = E[REFERENCE][0]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in solutions:
        if name != "Explicit Euler":
            axes[0].plot(tt, E[name] - e0, label=name)
    axes[0].set(title="Energy error, reference and symplectic-type methods", xlabel="t", ylabel="H(t) - H(0)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for name in (REFERENCE, "Explicit Euler"):
        if name in solutions:
            axes[1].plot(tt, E[name] - e0, label=name)
    axes[1].set(title="Energy error: explicit Euler", xlabel="t", ylabel="H(t) - H(0)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "arenstorf_energy.png"), dpi=150)
    plt.close(fig)
    print(f"Figures saved in {outdir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=300000,
                        help="number of steps over one period (300000 is the value used in the thesis)")
    parser.add_argument("--methods", nargs="*", default=NAMES[1:], choices=NAMES[1:],
                        help="methods to run besides the reference (default: all)")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args(argv)
    solutions = compare(args.n, args.methods)
    print_report(solutions, args.n)
    if not args.no_plots:
        save_figures(solutions, args.n, args.outdir)


if __name__ == "__main__":
    main()