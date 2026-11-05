"""Global optimisation of an Earth-Mars transfer with differential evolution.

Simplified model: Sun-centred two-body dynamics, circular coplanar planetary
orbits, two-impulse transfer solved as a Lambert problem. The decision variables
are the departure date and the time of flight; the cost is the sum of the
departure and arrival excess speeds. Units: AU, year, mu_sun = 4 pi^2.
"""
import os
import numpy as np
from scipy.optimize import differential_evolution

MU = 4 * np.pi**2          # AU^3 / yr^2
AU_PER_YR_KMS = 4.740470   # 1 AU/yr in km/s
A_EARTH, A_MARS = 1.0, 1.523679


def stumpff_c(z):
    if z > 1e-8:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < -1e-8:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 0.5 - z / 24.0


def stumpff_s(z):
    if z > 1e-8:
        s = np.sqrt(z)
        return (s - np.sin(s)) / s**3
    if z < -1e-8:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / s**3
    return 1.0 / 6.0 - z / 120.0


def lambert(r1, r2, tof, mu=MU):
    """Prograde Lambert solver (universal variables, bisection on z)."""
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.dot(r1, r2) / (n1 * n2)
    sign = 1.0 if (r1[0] * r2[1] - r1[1] * r2[0]) >= 0 else -1.0  # prograde
    dnu = np.arccos(np.clip(cos_dnu, -1, 1))
    if sign < 0:
        dnu = 2 * np.pi - dnu
    if abs(np.sin(dnu)) < 1e-4:   # transfer angle of 0 or 180 deg: Lambert is singular
        raise ValueError("degenerate transfer angle")
    a_par = np.sin(dnu) * np.sqrt(n1 * n2 / (1 - np.cos(dnu)))

    def y(z):
        return n1 + n2 + a_par * (z * stumpff_s(z) - 1) / np.sqrt(stumpff_c(z))

    def tof_of(z):
        yz = y(z)
        return ((yz / stumpff_c(z)) ** 1.5 * stumpff_s(z) + a_par * np.sqrt(yz)) / np.sqrt(mu)

    lo, hi = -4 * np.pi**2, 4 * np.pi**2
    # raise the lower bound until y(z) > 0
    while y(lo) < 0:
        lo += 0.5
    for _ in range(200):
        z = 0.5 * (lo + hi)
        if y(z) < 0 or tof_of(z) < tof:
            lo = z
        else:
            hi = z
    z = 0.5 * (lo + hi)
    yz = y(z)
    f = 1 - yz / n1
    g = a_par * np.sqrt(yz / mu)
    gdot = 1 - yz / n2
    v1 = (r2 - f * r1) / g
    v2 = (gdot * r2 - r1) / g
    return v1, v2


def planet(a, t, phase=0.0):
    """Position and velocity of a planet on a circular coplanar orbit."""
    n = np.sqrt(MU / a**3)
    th = n * t + phase
    r = a * np.array([np.cos(th), np.sin(th)])
    v = a * n * np.array([-np.sin(th), np.cos(th)])
    return r, v


def cost(x):
    """Sum of departure and arrival excess speeds (AU/yr)."""
    t0, tof = x
    r1, v1p = planet(A_EARTH, t0)
    r2, v2p = planet(A_MARS, t0 + tof)
    try:
        v1, v2 = lambert(r1, r2, tof)
    except (FloatingPointError, ZeroDivisionError, ValueError):
        return 1e3
    c = np.linalg.norm(v1 - v1p) + np.linalg.norm(v2 - v2p)
    return c if np.isfinite(c) else 1e3


def hohmann_cost():
    """Sum of the two excess speeds of a Hohmann transfer (AU/yr)."""
    v_e = np.sqrt(MU / A_EARTH)
    v_m = np.sqrt(MU / A_MARS)
    a_t = 0.5 * (A_EARTH + A_MARS)
    v_dep = np.sqrt(MU * (2 / A_EARTH - 1 / a_t))
    v_arr = np.sqrt(MU * (2 / A_MARS - 1 / a_t))
    return (v_dep - v_e) + (v_m - v_arr)


if __name__ == "__main__":
    t_syn = 1 / abs(1 - (A_EARTH / A_MARS) ** 1.5)   # synodic period (yr)
    bounds = [(0.0, t_syn), (0.3, 1.2)]
    res = differential_evolution(cost, bounds, seed=1, tol=1e-10, popsize=20,
                                 maxiter=300, polish=False)
    print(f"DE: t0 = {res.x[0]:.3f} yr, tof = {res.x[1]:.3f} yr, "
          f"cost = {res.fun * AU_PER_YR_KMS:.3f} km/s, evaluations = {res.nfev}")
    print(f"Hohmann: {hohmann_cost() * AU_PER_YR_KMS:.3f} km/s")

    # Pork-chop plot to show the landscape the optimiser is searching
    import matplotlib.pyplot as plt
    t0s = np.linspace(*bounds[0], 90)
    tofs = np.linspace(*bounds[1], 70)
    Z = np.array([[cost((a, b)) * AU_PER_YR_KMS for a in t0s] for b in tofs])
    Z = np.clip(Z, 0, 20)
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 5))
    cs = plt.contourf(t0s, tofs, Z, levels=np.linspace(5, 20, 31), cmap="viridis_r")
    plt.colorbar(cs, label="departure + arrival excess speed (km/s)")
    plt.plot(*res.x, "r*", ms=14, label="differential evolution")
    plt.xlabel("departure date (yr)")
    plt.ylabel("time of flight (yr)")
    plt.title("Earth-Mars transfer cost (circular coplanar model)")
    plt.legend()
    plt.savefig("figures/porkchop.png", dpi=150, bbox_inches="tight")
