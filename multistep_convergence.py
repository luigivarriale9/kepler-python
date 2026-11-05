import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from multistep import adams_bashforth, abm4_pece

e, T = 0.7, 3.0
y0 = np.array([1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])


def f(t, y):
    r3 = np.hypot(y[0], y[1]) ** 3
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


ref = solve_ivp(f, (0, T), y0, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]

steps = np.array([400, 800, 1600, 3200, 6400])
methods = {
    "AB2": lambda n: adams_bashforth(f, 0, T, y0, n, 2),
    "AB3": lambda n: adams_bashforth(f, 0, T, y0, n, 3),
    "AB4": lambda n: adams_bashforth(f, 0, T, y0, n, 4),
    "ABM4 (PECE)": lambda n: abm4_pece(f, 0, T, y0, n),
}

os.makedirs("figures", exist_ok=True)
plt.figure()
for name, run in methods.items():
    err = [np.linalg.norm(run(n)[1][-1] - ref) for n in steps]
    p = np.polyfit(np.log(T / steps), np.log(err), 1)[0]
    print(f"{name}: observed order {p:.2f}")
    plt.loglog(T / steps, err, "o-", label=f"{name} (order {p:.2f})")
plt.xlabel("step size h")
plt.ylabel("global error at t = T")
plt.title("Multistep methods on the Kepler problem (e = 0.7)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("figures/multistep_convergence.png", dpi=150, bbox_inches="tight")

for name in ("AB4", "ABM4 (PECE)"):
    print(name, "error at n = 1000:", np.linalg.norm(methods[name](1000)[1][-1] - ref))