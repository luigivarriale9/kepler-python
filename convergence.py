import numpy as np
import matplotlib.pyplot as plt
from kepler import (initial_state, integrate, reference, euler_step, symplectic_euler_step, midpoint_step_fsolve, verlet_step)

#Eulero esplicito entra nel regime asintotico solo con passi molto piccoli.
METHODS = {
    "Eulero esplicito": (euler_step, [20000, 40000, 80000, 160000]),
    "Eulero simplettico": (symplectic_euler_step, [500, 1000, 2000, 4000]),
    "MidPoint": (midpoint_step_fsolve, [500, 1000, 2000, 4000]),
    "Stormer-Verlet": (verlet_step, [500, 1000, 2000, 4000])
}

def global_errors(step, y0, t_end, ns):
    y_ref= reference(y0, t_end)
    return np.array([np.linalg.norm(integrate(step, y0, t_end, n)[-1] - y_ref) for n in ns])

def observed_order(hs, errs):
    """Pendenza della retta log(err)=p log(h) + c."""
    return np.polyfit(np.log(hs), np.log(errs), 1)[0]

if __name__== "__main__":
    y0, t_end =initial_state(0.7), 3.0
    plt.figure()
    for name, (step, ns) in METHODS.items():
        ns=np.array(ns)
        hs=t_end/ns
        errs=global_errors(step, y0, t_end, ns)
        print(f"{name}: ordine osservato = {observed_order(hs, errs):.2f}")
        plt.loglog(hs, errs, "o-", label=name)
    plt.xlabel("h")
    plt.ylabel("errore globale a t=3")
    plt.legend()
    plt.grid(True, which="both")
    plt.savefig("convergenza.png", dpi=150)
