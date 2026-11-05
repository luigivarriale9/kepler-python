import os

import numpy as np
import matplotlib.pyplot as plt
from kepler import (initial_state, integrate, reference, euler_step, symplectic_euler_step, midpoint_step_fsolve, verlet_step)

# Explicit Euler reaches the asymptotic regime only with very small steps.
METHODS = {
    "Explicit Euler": (euler_step, [20000, 40000, 80000, 160000]),
    "Symplectic Euler": (symplectic_euler_step, [500, 1000, 2000, 4000]),
    "Implicit midpoint": (midpoint_step_fsolve, [500, 1000, 2000, 4000]),
    "Störmer–Verlet": (verlet_step, [500, 1000, 2000, 4000])
}

def global_errors(step, y0, t_end, ns):
    """Global error at t_end for each number of steps in ns, against the reference solution."""
    y_ref= reference(y0, t_end)
    return np.array([np.linalg.norm(integrate(step, y0, t_end, n)[-1] - y_ref) for n in ns])

def observed_order(hs, errs):
    """Slope of the line log(err) = p log(h) + c."""
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]

if __name__== "__main__":
    y0, t_end =initial_state(0.7), 3.0
    plt.figure()
    for name, (step, ns) in METHODS.items():
        ns=np.array(ns)
        hs=t_end/ns
        errs=global_errors(step, y0, t_end, ns)
        print(f"{name}: observed order = {observed_order(hs, errs):.2f}")
        plt.loglog(hs, errs, "o-", label=name)
    plt.xlabel("h")
    plt.ylabel("global error at t = 3")
    plt.legend()
    plt.grid(True, which="both")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/convergence.png", dpi=150)