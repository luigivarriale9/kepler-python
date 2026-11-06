import os
import numpy as np
import matplotlib.pyplot as plt
from convergence import kepler_position
from multistep import adams_bashforth, adams_moulton, abm4_pece

e, T = 0.7, 3.0
y0 = np.array([1 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])


def f(t, y):
    r3 = np.hypot(y[0], y[1]) ** 3
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3])


q_exact = kepler_position(e, T)

steps = np.array([250, 500, 1000, 2000, 4000])
methods = {}
for k in range(1, 5):
    methods[f"AB{k}"] = lambda n, k=k: adams_bashforth(f, 0, T, y0, n, k)
for k in range(1, 5):
    methods[f"AM{k}"] = lambda n, k=k: adams_moulton(f, 0, T, y0, n, k)
methods["ABM4 (PECE)"] = lambda n: abm4_pece(f, 0, T, y0, n)

os.makedirs("figures", exist_ok=True)
plt.figure(figsize=(8, 6))
for name, run in methods.items():
    err = [np.linalg.norm(run(n)[1][-1][:2] - q_exact) for n in steps]
    print(name, " ".join(f"{x:.3e}" for x in err))
    p = np.polyfit(np.log(T / steps), np.log(err), 1)[0]
    print(f"{name}: observed order {p:.2f}")
    style = "--" if name.startswith("AM") else "-"
    plt.loglog(T / steps, err, "o" + style, ms=4, label=f"{name} (order {p:.2f})")
plt.xlabel("step size h")
plt.ylabel("global error at t = T")
plt.title("Multistep methods on the Kepler problem (e = 0.7)")
plt.legend(fontsize=8, ncol=2)
plt.grid(True, which="both", alpha=0.3)
plt.savefig("figures/multistep_convergence.png", dpi=150, bbox_inches="tight")

for name in ("AB4", "ABM4 (PECE)"):
    print(name, "error at n = 1000:", np.linalg.norm(methods[name](1000)[1][-1][:2] - q_exact))