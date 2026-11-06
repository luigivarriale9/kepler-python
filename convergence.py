import os

import numpy as np
import matplotlib.pyplot as plt
from kepler import (initial_state, integrate, reference, euler_step,
                    symplectic_euler_step, midpoint_step_fsolve, verlet_step)

E_ECC, T_END = 0.7, 3.0
NS = np.array([250, 500, 1000, 2000, 4000])   # same n as the MATLAB study

METHODS = {
    "Explicit Euler": euler_step,
    "Symplectic Euler": symplectic_euler_step,
    "Implicit midpoint": midpoint_step_fsolve,
    "Störmer–Verlet": verlet_step,
}


def kepler_position(e, t, tol=1e-14, maxit=50):
    """Exact position (a = 1, mu = 1, pericentre at t = 0) from Kepler's equation
    E - e sin E = M, M = t, solved with Newton."""
    E = t
    for _ in range(maxit):
        dE = (E - e * np.sin(E) - t) / (1 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return np.array([np.cos(E) - e, np.sqrt(1 - e**2) * np.sin(E)])


def global_errors(step, y0, t_end, ns, q_exact):
    """Position error at t_end for each n in ns, against the exact solution."""
    return np.array([np.linalg.norm(integrate(step, y0, t_end, n)[-1][:2] - q_exact)
                     for n in ns])


def observed_order(hs, errs):
    """Slope of the line log(err) = p log(h) + c."""
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]


if __name__ == "__main__":
    y0 = initial_state(E_ECC)
    q_exact = kepler_position(E_ECC, T_END)

    # independent check: the DOP853 reference must agree with the exact solution
    print(f"|DOP853 - exact| at t = {T_END}: "
          f"{np.linalg.norm(reference(y0, T_END)[:2] - q_exact):.2e}")

    hs = T_END / NS
    table = {}
    plt.figure()
    for name, step in METHODS.items():
        errs = global_errors(step, y0, T_END, NS, q_exact)
        table[name] = errs
        print(f"{name}: observed order = {observed_order(hs, errs):.2f}")
        plt.loglog(hs, errs, "o-", label=name)

    print("\nn      " + "  ".join(f"{n:>10d}" for n in NS))
    for name, errs in table.items():
        print(f"{name[:16]:<16}" + "  ".join(f"{e:10.3e}" for e in errs))

    plt.xlabel("h")
    plt.ylabel("global error at t = 3")
    plt.legend()
    plt.grid(True, which="both")
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/convergence.png", dpi=150)