import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kepler import initial_state, rhs, euler_step, symplectic_euler_step, verlet_step
from variational import (METHODS_AUG, initial_augmented, midpoint_step_aug, propagate,
                         reference_stm, symplecticity_error, unpack)

P = 2 * np.pi          # orbital period (a = 1, mu = 1)
X0 = initial_state(0.7)


def test_reference_stm_matches_finite_differences():
    T = P
    flow = lambda x: solve_ivp(rhs, (0, T), x, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
    eps = 1e-6
    fd = np.column_stack([(flow(X0 + eps * e) - flow(X0 - eps * e)) / (2 * eps) for e in np.eye(4)])
    _, Phi = reference_stm(X0, T)
    assert np.linalg.norm(fd - Phi) / np.linalg.norm(Phi) < 1e-4


def _midpoint_state_map(x, h):
    return unpack(midpoint_step_aug(initial_augmented(x), h))[0]


@pytest.mark.parametrize("name, state_map", [
    ("Explicit Euler", euler_step),
    ("Symplectic Euler", symplectic_euler_step),
    ("Störmer–Verlet", verlet_step),
    ("Implicit midpoint", _midpoint_state_map),
])
def test_one_step_jacobian_matches_finite_differences(name, state_map):
    """The matrix propagated with each scheme is the Jacobian of the numerical map."""
    x, h, eps = np.array([0.4, 0.3, -0.5, 1.7]), 0.01, 1e-6
    fd = np.column_stack([(state_map(x + eps * e, h) - state_map(x - eps * e, h)) / (2 * eps)
                         for e in np.eye(4)])
    _, Phi = unpack(METHODS_AUG[name](initial_augmented(x), h))
    assert np.linalg.norm(fd - Phi) < 1e-7


@pytest.mark.parametrize("name", ["Symplectic Euler", "Implicit midpoint", "Störmer–Verlet"])
def test_symplectic_methods_preserve_phi_J_phi(name):
    t, W = propagate(METHODS_AUG[name], X0, 5 * P, 1000)
    _, Phi = unpack(W[-1])
    assert symplecticity_error(Phi) < 1e-7
    assert abs(np.linalg.det(Phi) - 1) < 1e-7


def test_non_symplectic_methods_do_not():
    err = {}
    for name in ("Explicit Euler", "RK4"):
        t, W = propagate(METHODS_AUG[name], X0, 5 * P, 1000)
        err[name] = symplecticity_error(unpack(W[-1])[1])
    t, W = propagate(METHODS_AUG["Störmer–Verlet"], X0, 5 * P, 1000)
    err_v = symplecticity_error(unpack(W[-1])[1])
    assert err["RK4"] > 1e-3 and err["RK4"] > 1e4 * err_v
    assert err["Explicit Euler"] > 1.0


def test_verlet_stm_has_second_order():
    T = 3 * P
    _, Pr = reference_stm(X0, T)
    errs = []
    for n in (3000, 6000, 12000):
        _, W = propagate(METHODS_AUG["Störmer–Verlet"], X0, T, n)
        errs.append(np.linalg.norm(unpack(W[-1])[1] - Pr) / np.linalg.norm(Pr))
    p = np.log2(errs[1] / errs[2])
    assert 1.7 < p < 2.3, p
