import numpy as np
from kepler import (initial_state, integrate, energy, angular_momentum, euler_step, symplectic_euler_step, midpoint_step_fsolve, verlet_step, reference)
from convergence import global_errors, observed_order, kepler_position

def test_initial_state_has_expected_energy_and_momentum():
    y0 = initial_state(0.7)
    assert np.isclose(energy(y0), -0.5) # E = -mu / (2a), a=1
    assert np.isclose(angular_momentum(y0), np.sqrt(1-0.7**2))

def test_verlet_has_second_order():
    y0, t_end = initial_state(0.7), 3.0
    ns = np.array([500, 1000, 2000])
    q_exact = kepler_position(0.7, t_end)
    p = observed_order(t_end / ns, global_errors(verlet_step, y0, t_end, ns, q_exact))
    assert abs(p - 2) < 0.25

def test_symplectic_euler_has_first_order():
    y0, t_end = initial_state(0.7), 3.0
    ns = np.array([500, 1000, 2000])
    q_exact = kepler_position(0.7, t_end)
    p = observed_order(t_end / ns, global_errors(symplectic_euler_step, y0, t_end, ns, q_exact))
    assert 0.8 < p < 1.2

def test_angular_momentum_is_conserved_by_verlet():
    y0 = initial_state(0.7)
    ys = integrate(verlet_step, y0, 20 * np.pi, 2000)
    L = np.array([angular_momentum(y) for y in ys])
    assert np.abs(L - L[0]).max() < 1e-10
    
def test_verlet_energy_error_stays_bounded():
    y0 = initial_state(0.7)
    ys = integrate(verlet_step, y0, 100 * 2 * np.pi, 100 * 2000)
    E = np.array([energy(y) for y in ys])
    assert np.abs(E - E[0]).max() < 1e-3

def test_dop853_reference_matches_exact_solution():
    y0, t_end = initial_state(0.7), 3.0
    assert np.linalg.norm(reference(y0, t_end)[:2] - kepler_position(0.7, t_end)) < 1e-10