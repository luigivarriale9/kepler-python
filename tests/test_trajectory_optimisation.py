import numpy as np
from astropy.time import Time
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from trajectory_optimisation import (A_EARTH, A_MARS, A_VENUS, KMS_PER_AUDAY, MU, MU_DAY, cost, hohmann_cost, lambert, optimise_ephemeris, planet, planet_spline)


def test_lambert_reproduces_target_position():
    """Propagating the Lambert initial velocity must reach r2 after the time of flight."""
    t0, tof = 0.3, 0.6
    r1, _ = planet(A_EARTH, t0)
    r2, _ = planet(A_MARS, t0 + tof)
    v1, v2 = lambert(r1, r2, tof)

    def rhs(t, y):
        return np.r_[y[2:], -MU * y[:2] / np.linalg.norm(y[:2]) ** 3]

    sol = solve_ivp(rhs, (0, tof), np.r_[r1, v1], rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(sol.y[:2, -1] - r2) < 1e-6
    assert np.linalg.norm(sol.y[2:, -1] - v2) < 1e-6


def test_differential_evolution_finds_hohmann_cost():
    """The global minimum of the circular coplanar problem is the Hohmann transfer."""
    t_syn = 1 / abs(1 - (A_EARTH / A_MARS) ** 1.5)
    res = differential_evolution(cost, [(0.0, t_syn), (0.3, 1.2)], seed=1,
                                 popsize=20, maxiter=300, tol=1e-10, polish=False)
    assert abs(res.fun - hohmann_cost()) / hohmann_cost() < 0.01

def test_differential_evolution_finds_hohmann_cost_for_venus():
    t_syn = 1 / abs(1 - (A_EARTH / A_VENUS) ** 1.5)
    res = differential_evolution(cost, [(0.0, t_syn), (0.2, 1.0)], args=(A_VENUS,),
                                 seed=1, popsize=20, maxiter=300, tol=1e-10, polish=False)
    assert abs(res.fun - hohmann_cost(A_VENUS)) / hohmann_cost(A_VENUS) < 0.01

def test_earth_state_is_realistic():
    """Earth is about 1 AU from the Sun and moves at about 30 km/s."""
    jd = Time("2026-11-07").jd
    earth = planet_spline("earth", jd, jd + 5)
    assert 0.98 < np.linalg.norm(earth(jd)) < 1.02
    assert 29.0 < np.linalg.norm(earth.derivative()(jd)) * KMS_PER_AUDAY < 30.6


def test_lambert_3d_reaches_target():
    """Propagating the Lambert velocity in 3D must reach the target position."""
    jd = Time("2026-11-07").jd
    earth = planet_spline("earth", jd, jd + 400)
    mars = planet_spline("mars", jd, jd + 400)
    tof = 300.0
    r1, r2 = earth(jd), mars(jd + tof)
    v1, _ = lambert(r1, r2, tof, mu=MU_DAY)

    def rhs(t, y):
        return np.r_[y[3:], -MU_DAY * y[:3] / np.linalg.norm(y[:3]) ** 3]

    sol = solve_ivp(rhs, (0, tof), np.r_[r1, v1], rtol=1e-11, atol=1e-13)
    assert np.linalg.norm(sol.y[:3, -1] - r2) < 1e-6


def test_earth_mars_2026_window_cost_is_plausible():
    """The real-ephemeris optimum is close to the circular-model Hohmann cost (5.59 km/s)."""
    res, _ = optimise_ephemeris("mars", "2026-10-01", "2027-12-01")
    assert 5.3 < res.fun * KMS_PER_AUDAY < 6.2