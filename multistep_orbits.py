import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multistep import adams_bashforth, abm4_pece

e = 0.7
periods, n = 2, 1000  # coarse step on purpose, to make the errors visible
T = periods * 2 * np.pi
y0 = np.array([1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])


def f(t, y):
    r3 = np.hypot(y[0], y[1]) ** 3
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


ref = solve_ivp(f, (0, T), y0, method="DOP853", rtol=1e-13, atol=1e-13,
                dense_output=True)
t_ref = np.linspace(0, T, 2000)
y_ref = ref.sol(t_ref)

runs = {
    "AB2": adams_bashforth(f, 0, T, y0, n, 2)[1],
    "AB4": adams_bashforth(f, 0, T, y0, n, 4)[1],
    "ABM4 (PECE)": abm4_pece(f, 0, T, y0, n)[1],
}

os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(1, len(runs), figsize=(13, 4.5), sharex=True, sharey=True)
for ax, (name, y) in zip(axes, runs.items()):
    err = np.linalg.norm(y[-1] - ref.y[:, -1])
    ax.plot(y_ref[0], y_ref[1], "k-", lw=1, label="DOP853 (reference)")
    ax.plot(y[:, 0], y[:, 1], "r-", lw=0.8, label=name)
    ax.plot(0, 0, "ko", ms=4)
    ax.set_title(f"{name}, error at final time: {err:.1e}")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.legend(loc="lower left", fontsize=8)
axes[0].set_ylabel("y")
fig.suptitle(f"Kepler orbit, e = {e}, {periods} periods, {n} steps")
fig.tight_layout()
plt.savefig("figures/multistep_orbits.png", dpi=150, bbox_inches="tight")