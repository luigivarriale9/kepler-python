"""Integrators on a test trajectory of the planar circular restricted three-body problem.

Transposed from symplectic_integrators/three_body_problem.m of the MATLAB repository.
Test case (mu = 1.215e-2, initial state) from Rubinsztejn's Matlab-Astrodynamics-Library;
the initial velocity can be rescaled (same direction) to a chosen Jacobi constant C.
The MATLAB state has six variables, but this test case has z = vz = 0, so z stays zero and
the planar model of pcr3bp.py gives the same trajectory.

Methods: RK45 with default tolerances and with rtol = atol = 1e-12 (the counterpart of
MATLAB ode45), implicit midpoint, explicit Euler, Stormer-Verlet, symplectic Euler.
Figures: the trajectory of each method over the forbidden region and the Lagrange points,
and the change of the Jacobi constant.

Usage:
    python3 three_body_comparison.py [--c 3.1624] [--methods NAME ...] [--no-plots]
"""
import argparse
import os
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp

from kepler import euler_step, integrate, midpoint_step_fsolve, symplectic_euler_step, verlet_step
from pcr3bp import (acceleration, jacobi, lagrange_points, rescale_to_jacobi, rhs,
                    zero_velocity_jacobi)

MU = 1.215e-2
Y0 = np.array([1 - MU, 0.0455, -0.5322, 0.2])     # (x, y, vx, vy)
T_END = 20.0
XTOL = 1e-12        # tolerance of the implicit solves, as in the MATLAB code
NAMES = ["ODE45 default", "Implicit midpoint", "Explicit Euler", "ODE45 tight",
         "Stormer-Verlet", "Symplectic Euler"]
STEPS = {"Implicit midpoint": 15394, "Stormer-Verlet": 15394,
         "Symplectic Euler": 120000, "Explicit Euler": 700000}


def solve(name, y0, t_end=T_END, mu=MU, steps=STEPS):
    """Return (t, Y) of the given method; Y has shape (len(t), 4)."""
    f = partial(rhs, mu=mu)
    if name in ("ODE45 default", "ODE45 tight"):
        tol = {} if name == "ODE45 default" else {"rtol": 1e-12, "atol": 1e-12}
        sol = solve_ivp(f, (0.0, t_end), y0, method="RK45", **tol)
        return sol.t, sol.y.T

    acc = partial(acceleration, mu=mu)

    step = {
        "Implicit midpoint": partial(midpoint_step_fsolve, f=f, xtol=XTOL),
        "Explicit Euler": partial(euler_step, f=f),
        "Stormer-Verlet": partial(verlet_step, acc=acc),
        "Symplectic Euler": partial(symplectic_euler_step, acc=acc, implicit=True),
    }[name]
    n = steps[name]
    return np.linspace(0.0, t_end, n + 1), integrate(step, y0, t_end, n)


def run(y0, names, steps=STEPS, mu=MU):
    """Return {name: (t, Y)} for the requested methods."""
    return {name: solve(name, y0, mu=mu, steps=steps) for name in names}


def print_report(solutions, c, mu=MU):
    print(f"Jacobi constant used: C = {c:.6f}")
    print("Change of the Jacobi constant, max |dC| and final value:")
    for name, (t, Y) in solutions.items():
        dc = jacobi(Y, mu) - jacobi(Y[0], mu)
        print(f"  {name:18s} {np.max(np.abs(dc)):.3e}  {dc[-1]:+.3e}   ({len(t) - 1} steps)")


def save_figures(solutions, c, outdir, mu=MU, h=0.005):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    g = np.arange(-1.5, 1.5 + h / 2, h)
    X, Yg = np.meshgrid(g, g)
    Cgrid = zero_velocity_jacobi(X, Yg, mu)
    L = lagrange_points(mu)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, (name, (t, Y)) in zip(axes.ravel(), solutions.items()):
        ax.contourf(X, Yg, Cgrid, levels=[0, c], colors=["0.85"])
        ax.contour(X, Yg, Cgrid, levels=[c], colors="k", linewidths=0.6)
        ax.plot(Y[:, 0], Y[:, 1], "b", lw=0.7)
        ax.plot(L[:, 0], L[:, 1], "kx", ms=6)
        ax.plot(-mu, 0, "o", color="tab:red", ms=6)
        ax.plot(1 - mu, 0, "o", color="tab:blue", ms=5)
        ax.set(title=name, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), aspect="equal", xlabel="q1", ylabel="q2")
    for ax in axes.ravel()[len(solutions):]:
        ax.axis("off")
    fig.suptitle(f"Trajectories over the forbidden region (grey), C = {c:.4f}, T = {T_END:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cr3bp_trajectories.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for name, (t, Y) in solutions.items():
        ax.plot(t, jacobi(Y, mu) - jacobi(Y[0], mu), lw=1.0, label=name)
    ax.set(title="Change of the Jacobi constant", xlabel="Time (non-dimensional)", ylabel="Delta C")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "cr3bp_jacobi_drift.png"), dpi=150)
    plt.close(fig)
    print(f"Figures saved in {outdir}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--c", type=float, default=None,
                        help="Jacobi constant (default: the one of the initial state)")
    parser.add_argument("--methods", nargs="*", default=NAMES, choices=NAMES)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--steps-scale", type=float, default=1.0,
                        help="multiplies the number of steps of the fixed-step methods")
    args = parser.parse_args(argv)
    y0 = Y0 if args.c is None else rescale_to_jacobi(Y0, args.c, MU)
    steps = {k: max(10, int(v * args.steps_scale)) for k, v in STEPS.items()}
    solutions = run(y0, args.methods, steps)
    c = jacobi(y0, MU)
    print_report(solutions, c)
    if not args.no_plots:
        save_figures(solutions, c, args.outdir)


if __name__ == "__main__":
    main()