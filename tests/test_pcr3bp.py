import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pcr3bp import (MU, T_ARENSTORF, Y0_ARENSTORF, acceleration, from_canonical, hamiltonian,
                    jacobi, lagrange_points, rescale_to_jacobi, rhs, to_canonical,
                    zero_velocity_jacobi)


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


def _random_states(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(-1.2, 1.2, n), rng.uniform(0.1, 1.0, n),
                            rng.normal(size=n), rng.normal(size=n)])


def test_acceleration_is_the_acceleration_part_of_rhs():
    for s in _random_states():
        assert np.array_equal(acceleration(s[:2], s[2:]), rhs(0.0, s)[2:])


def test_jacobi_is_minus_twice_the_hamiltonian_for_one_state_and_for_an_array():
    S = _random_states()
    expected = [-2 * hamiltonian(to_canonical(s)) for s in S]
    assert np.allclose(jacobi(S), expected, atol=1e-12)
    assert np.ndim(jacobi(S[0])) == 0


def test_zero_velocity_jacobi_is_jacobi_of_a_state_at_rest():
    x, y = 0.3, -0.4
    assert zero_velocity_jacobi(x, y) == pytest.approx(jacobi(np.array([x, y, 0.0, 0.0])), abs=1e-14)


def test_rescaling_gives_the_requested_constant_and_keeps_position_and_direction():
    s = np.array([0.9, 0.0455, -0.5322, 0.2])
    out = rescale_to_jacobi(s, 3.0)
    assert jacobi(out) == pytest.approx(3.0, abs=1e-12)
    assert np.array_equal(out[:2], s[:2])
    assert abs(out[2] * s[3] - out[3] * s[2]) < 1e-12


def test_rescaling_refuses_a_constant_with_no_real_velocity():
    s = np.array([0.9, 0.0455, -0.5322, 0.2])
    with pytest.raises(ValueError):
        rescale_to_jacobi(s, zero_velocity_jacobi(s[0], s[1]) + 0.1)


def test_lagrange_points_are_equilibria_and_l4_is_equilateral():
    for mu in (MU, 1.215e-2):
        L = lagrange_points(mu)
        for x, y in L:
            assert np.max(np.abs(rhs(0.0, np.array([x, y, 0.0, 0.0]), mu)[2:])) < 1e-9
        assert np.hypot(L[3, 0] + mu, L[3, 1]) == pytest.approx(1.0)
        assert np.hypot(L[3, 0] - 1 + mu, L[3, 1]) == pytest.approx(1.0)