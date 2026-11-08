import numpy as np
from scipy.integrate import solve_ivp

from pcr3bp import (MU, T_ARENSTORF, Y0_ARENSTORF, from_canonical, hamiltonian, jacobi,
                   rhs, to_canonical)


def _reference():
    return solve_ivp(rhs, (0, T_ARENSTORF), Y0_ARENSTORF, method="DOP853",
                     rtol=1e-12, atol=1e-12)


def test_canonical_roundtrip():
    s = np.array([0.3, -0.2, 0.7, 0.1])
    assert np.allclose(from_canonical(to_canonical(s)), s)


def test_hamiltonian_equals_energy_in_velocity_variables():
    """H(q, p) must equal v^2/2 - r^2/2 - (1-mu)/r1 - mu/r2 written with the velocities."""
    x, y, vx, vy = Y0_ARENSTORF
    r1, r2 = np.hypot(x + MU, y), np.hypot(x - 1 + MU, y)
    h_vel = 0.5 * (vx**2 + vy**2) - 0.5 * (x**2 + y**2) - (1 - MU) / r1 - MU / r2
    assert abs(hamiltonian(to_canonical(Y0_ARENSTORF)) - h_vel) < 1e-13


def test_arenstorf_orbit_is_periodic():
    """After one period the state returns to the initial one (limited by the digits of the data)."""
    sol = _reference()
    assert np.linalg.norm(sol.y[:, -1] - Y0_ARENSTORF) < 1e-6


def test_jacobi_constant_is_conserved_along_the_orbit():
    sol = _reference()
    c = np.array([jacobi(sol.y[:, i]) for i in range(sol.y.shape[1])])
    assert c.max() - c.min() < 1e-9
