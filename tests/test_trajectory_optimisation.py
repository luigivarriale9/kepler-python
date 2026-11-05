import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from trajectory_optimisation import (A_EARTH, A_MARS, MU, cost, hohmann_cost,
                                     lambert, planet)


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
