import os
import numpy as np
import matplotlib.pyplot as plt
from multistep import adams_bashforth, adams_moulton, abm4_pece
from convergence import kepler_position

e = 0.7
periods, n = 2, 1000  # coarse step on purpose, to make the errors visible
T = periods * 2 * np.pi
y0 = np.array([1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])


def f(t, y):
    r3 = np.hypot(y[0], y[1]) ** 3
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


t_ref = np.linspace(0, T, 2000)
q_ref = np.array([kepler_position(e, t) for t in t_ref]).T   # exact positions, shape (2, 2000)
q_end = kepler_position(e, T)

runs = {f"AB{k}": adams_bashforth(f, 0, T, y0, n, k)[1] for k in range(1, 5)}
runs.update({f"AM{k}": adams_moulton(f, 0, T, y0, n, k)[1] for k in range(1, 5)})
runs["ABM4 (PECE)"] = abm4_pece(f, 0, T, y0, n)[1]

os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharex=True, sharey=True)
for ax, (name, y) in zip(axes.flat, runs.items()):
    err = np.linalg.norm(y[-1][:2] - q_end)
    ax.plot(q_ref[0], q_ref[1], "k-", lw=1, label="exact solution")
    ax.plot(y[:, 0], y[:, 1], "r-", lw=0.8, label=name)
    ax.plot(0, 0, "ko", ms=4)
    ax.set_title(f"{name}, error at final time: {err:.1e}")
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
fig.suptitle(f"Kepler orbit, e = {e}, {periods} periods, {n} steps")
fig.tight_layout()
plt.savefig("figures/multistep_orbits.png", dpi=130, bbox_inches="tight")