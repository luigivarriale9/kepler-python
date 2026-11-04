import numpy as np
import matplotlib.pyplot as plt
from kepler import (initial_state, integrate, euler_step, symplectic_euler_step, midpoint_step_fsolve, verlet_step)

METHODS = {
    "Eulero esplicito": euler_step,
    "Eulero simplettico": symplectic_euler_step,
    "MidPoint": midpoint_step_fsolve,
    "Stormer-Verlet": verlet_step,
}

def exact_orbit(e, n=400):
    """Ellisse di Keplero con a=1, parametrizzata dall'anomalia vera."""
    theta = np.linspace(0, 2 * np.pi, n)
    r = (1-e**2) / (1 + e * np.cos(theta))
    return r * np.cos(theta), r * np.sin(theta)

if __name__ == "__main__":
    e = 0.7
    n_orbits = 5
    steps_per_orbit = 200
    y0 = initial_state(e)
    t_end = n_orbits * 2 * np.pi
    n_steps = n_orbits * steps_per_orbit
    xe, ye = exact_orbit(e)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, step) in zip(axes.flat, METHODS.items()):
        ys = integrate(step, y0, t_end, n_steps)
        ax.plot(ys[:, 0], ys[:, 1], lw=0.8, label=name)
        ax.plot(xe, ye, "k--", lw=0.8, label="orbita esatta")
        ax.plot(0, 0, "ko")
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(-2.5, 1.5)
        ax.set_ylim(-2, 2)
        ax.grid(True)
        ax.legend(loc="upper right")

    fig.suptitle(f"Orbite di Keplero, e={e}, {n_orbits} orbite, {steps_per_orbit} passi per orbita")
    plt.savefig("orbite.png", dpi=150)