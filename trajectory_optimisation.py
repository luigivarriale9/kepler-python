"""Global optimisation of an Earth-planet transfer with differential evolution.

Two models are available. Circular model: Sun-centred two-body dynamics, circular
coplanar planetary orbits (units: AU, year, mu_sun = 4 pi^2). Ephemeris model: real
planetary positions from the astropy built-in ephemeris, 3D transfer (units: AU, day).
In both, the transfer is a two-impulse Lambert problem, the decision variables are the
departure date and the time of flight, and the cost is the sum of the departure and
arrival excess speeds. Destinations: Mars, Venus.
"""
import argparse
import os
import warnings

import numpy as np
from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from erfa import ErfaWarning
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import differential_evolution

MU = 4 * np.pi**2          # AU^3 / yr^2
AU_PER_YR_KMS = 4.740470   # 1 AU/yr in km/s
A_EARTH, A_MARS, A_VENUS = 1.0, 1.523679, 0.723332

A_PLANET = {"mars": A_MARS, "venus": A_VENUS}
TOF_YR = {"mars": (0.3, 1.2), "venus": (0.2, 1.0)}            # circular model: bounds (yr)

# Real-ephemeris model (units: AU, day)
MU_DAY = 2.9591220828559e-4    # AU^3 / day^2 (Gaussian constant squared)
KMS_PER_AUDAY = 1731.4568      # 1 AU/day in km/s
EPS = np.radians(23.4392911)   # obliquity of the ecliptic (J2000)
TOF_DAYS = {"mars": (120.0, 420.0), "venus": (80.0, 300.0)}   # ephemeris model: bounds (days)
warnings.filterwarnings("ignore", category=ErfaWarning)
solar_system_ephemeris.set("builtin")

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


def cost(x, a_target=A_MARS):
    """Sum of departure and arrival excess speeds (AU/yr)."""
    t0, tof = x
    r1, v1p = planet(A_EARTH, t0)
    r2, v2p = planet(a_target, t0 + tof)
    try:
        v1, v2 = lambert(r1, r2, tof)
    except (FloatingPointError, ZeroDivisionError, ValueError):
        return 1e3
    c = np.linalg.norm(v1 - v1p) + np.linalg.norm(v2 - v2p)
    return c if np.isfinite(c) else 1e3


def hohmann_cost(a_target=A_MARS):
    """Sum of the two excess speeds of a Hohmann transfer (AU/yr), outer or inner target."""
    v_e = np.sqrt(MU / A_EARTH)
    v_m = np.sqrt(MU / a_target)
    a_t = 0.5 * (A_EARTH + a_target)
    v_dep = np.sqrt(MU * (2 / A_EARTH - 1 / a_t))
    v_arr = np.sqrt(MU * (2 / a_target - 1 / a_t))
    return abs(v_dep - v_e) + abs(v_m - v_arr)

def to_ecliptic(a):
    """Rotate ICRS (equatorial) vectors, shape (3, N), to the ecliptic frame."""
    c, s = np.cos(EPS), np.sin(EPS)
    return np.array([a[0], c * a[1] + s * a[2], -s * a[1] + c * a[2]])


def planet_spline(body, jd0, jd1):
    """Heliocentric position (AU) of `body` as a cubic Hermite spline in Julian date;
    the derivative of the spline is the velocity (AU/day)."""
    jd = np.arange(jd0, jd1 + 1.0, 1.0)
    t = Time(jd, format="jd", scale="tdb")
    p, v = get_body_barycentric_posvel(body, t)
    ps, vs = get_body_barycentric_posvel("sun", t)
    r = to_ecliptic((p.xyz - ps.xyz).to("au").value)
    vel = to_ecliptic((v.xyz - vs.xyz).to("au/day").value)
    return CubicHermiteSpline(jd, r.T, vel.T)


def ephemeris_cost(target, jd0, jd1, tof_max):
    """Cost function (AU/day) of an Earth -> target transfer with real ephemerides.
    The decision variables are the departure Julian date and the time of flight (days)."""
    earth = planet_spline("earth", jd0, jd1 + tof_max)
    dest = planet_spline(target, jd0, jd1 + tof_max)
    earth_v, dest_v = earth.derivative(), dest.derivative()

    def cost_eph(x):
        jd, tof = x
        r1, v1p = earth(jd), earth_v(jd)
        r2, v2p = dest(jd + tof), dest_v(jd + tof)
        try:
            v1, v2 = lambert(r1, r2, tof, mu=MU_DAY)
        except (FloatingPointError, ZeroDivisionError, ValueError):
            return 1e3
        c = np.linalg.norm(v1 - v1p) + np.linalg.norm(v2 - v2p)
        return c if np.isfinite(c) else 1e3

    return cost_eph


def optimise_ephemeris(target, start, end, seed=1):
    """Differential evolution on (departure date, time of flight) with real ephemerides."""
    jd0, jd1 = Time(start).jd, Time(end).jd
    tof_lo, tof_hi = TOF_DAYS[target]
    cost_eph = ephemeris_cost(target, jd0, jd1, tof_hi)
    res = differential_evolution(cost_eph, [(jd0, jd1), (tof_lo, tof_hi)], seed=seed,
                                 popsize=40, maxiter=400, tol=1e-10, polish=False)
    return res, cost_eph


def porkchop(xs, ys, Z, best, xlabel, ylabel, title, filename):
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(7, 5))
    cs = plt.contourf(xs, ys, np.clip(Z, 0, 20), levels=np.linspace(5, 20, 31), cmap="viridis_r")
    plt.colorbar(cs, label="departure + arrival excess speed (km/s)")
    plt.plot(*best, "r*", ms=14, label="differential evolution")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Earth -> planet transfer with differential evolution")
    parser.add_argument("target", choices=sorted(A_PLANET))
    parser.add_argument("--model", choices=["ephemeris", "circular"], default="ephemeris",
                        help="real ephemerides (default) or circular coplanar orbits")
    parser.add_argument("--start", default="2026-10-01", help="first departure date (ephemeris model)")
    parser.add_argument("--end", default="2028-12-01", help="last departure date (ephemeris model)")
    args = parser.parse_args()
    name = args.target.capitalize()

    if args.model == "circular":
        a = A_PLANET[args.target]
        t_syn = 1 / abs(1 - (A_EARTH / a) ** 1.5)   # synodic period (yr)
        bounds = [(0.0, t_syn), TOF_YR[args.target]]
        res = differential_evolution(cost, bounds, args=(a,), seed=1, tol=1e-10, popsize=20,
                                     maxiter=300, polish=False)
        print(f"Earth -> {args.target}, circular model: t0 = {res.x[0]:.3f} yr, "
              f"tof = {res.x[1]:.3f} yr, cost = {res.fun * AU_PER_YR_KMS:.3f} km/s, "
              f"evaluations = {res.nfev}")
        print(f"Hohmann: {hohmann_cost(a) * AU_PER_YR_KMS:.3f} km/s")
        t0s = np.linspace(*bounds[0], 90)
        tofs = np.linspace(*bounds[1], 70)
        Z = np.array([[cost((u, w), a) * AU_PER_YR_KMS for u in t0s] for w in tofs])
        porkchop(t0s, tofs, Z, res.x, "departure date (yr)", "time of flight (yr)",
                 f"Earth-{name} transfer cost (circular coplanar model)",
                 f"figures/porkchop_{args.target}_circular.png")
    else:
        res, cost_eph = optimise_ephemeris(args.target, args.start, args.end)
        print(f"Earth -> {args.target}, real ephemerides: departure {Time(res.x[0], format='jd').iso[:10]}, "
              f"time of flight {res.x[1]:.0f} d, arrival {Time(res.x[0] + res.x[1], format='jd').iso[:10]}, "
              f"cost = {res.fun * KMS_PER_AUDAY:.3f} km/s, evaluations = {res.nfev}")
        jd0, jd1 = Time(args.start).jd, Time(args.end).jd
        jds = np.linspace(jd0, jd1, 140)
        tofs = np.linspace(*TOF_DAYS[args.target], 90)
        Z = np.array([[cost_eph((u, w)) * KMS_PER_AUDAY for u in jds] for w in tofs])
        porkchop(jds - jd0, tofs, Z, (res.x[0] - jd0, res.x[1]), f"days after {args.start}",
                 "time of flight (days)", f"Earth-{name} transfer cost (real ephemerides)",
                 f"figures/porkchop_{args.target}_ephemeris.png")