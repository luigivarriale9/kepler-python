"""Tests of the generic arguments of the one-step methods in kepler.py: f=, acc=, implicit=.

Small problems with a known answer: y' = lam*y, the harmonic oscillator and a damped
oscillator (the damping makes the acceleration depend on the velocity, as the Coriolis
term does in the rotating-frame three-body problem).
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kepler import (acceleration, euler_step, midpoint_step_fsolve,
                    symplectic_euler_step, verlet_step)

ZETA = 0.1
Y0 = np.array([1.0, 0.0])      # (q, p) of the one-dimensional oscillators
T_END = 1.0
NS = [100, 200, 400, 800]


def harmonic(q, v):
    return -q


def damped(q, v):
    return -q - 2 * ZETA * v


def observed_order(step, ref, **kwargs):
    """Slope of log(error) against log(h) for the step applied over [0, T_END]."""
    errors = []
    for n in NS:
        h, y = T_END / n, Y0.copy()
        for _ in range(n):
            y = step(y, h, **kwargs)
        errors.append(np.linalg.norm(y - ref))
    return np.polyfit(np.log(T_END / np.array(NS)), np.log(errors), 1)[0]


@pytest.fixture(scope="module")
def references():
    exact_harmonic = np.array([np.cos(T_END), -np.sin(T_END)])
    sol = solve_ivp(lambda t, y: [y[1], damped(y[:1], y[1:])[0]], (0, T_END), Y0,
                    method="DOP853", rtol=1e-13, atol=1e-13)
    return {"harmonic": exact_harmonic, "damped": sol.y[:, -1]}


def test_acceleration_ignores_the_velocity():
    q = np.array([1.0, 0.5])
    assert np.array_equal(acceleration(q), acceleration(q, np.array([3.0, 4.0])))


def test_euler_uses_the_given_right_hand_side():
    lam, h = -2.0, 0.1
    y_new = euler_step(np.array([1.0]), h, f=lambda t, y: lam * y)
    assert abs(y_new[0] - (1 + lam * h)) < 1e-14


def test_midpoint_uses_the_given_right_hand_side():
    lam, h = -2.0, 0.1
    y_new = midpoint_step_fsolve(np.array([1.0]), h, f=lambda t, y: lam * y)
    assert abs(y_new[0] - (1 + lam * h / 2) / (1 - lam * h / 2)) < 1e-10
    
def test_midpoint_accepts_a_tolerance_for_the_implicit_solve():
    from kepler import initial_state, rhs
    y, h = initial_state(0.7), 0.01
    z = midpoint_step_fsolve(y, h, xtol=1e-12)
    assert np.linalg.norm(z - y - h * rhs(0.0, 0.5 * (y + z))) < 1e-12

def test_verlet_is_second_order_when_acc_depends_only_on_q(references):
    assert 1.9 < observed_order(verlet_step, references["harmonic"], acc=harmonic) < 2.1


def test_verlet_is_first_order_when_acc_depends_on_the_velocity(references):
    """The scheme is not symmetric if the velocity enters acc (documented in the docstring)."""
    assert 0.9 < observed_order(verlet_step, references["damped"], acc=damped) < 1.1


def test_verlet_step_matches_the_formula_when_acc_depends_on_the_velocity():
    """Half step in p with the old velocity, full step in q, half step in p with the
    half-step velocity (the transposition of fp(g, q_new) of the MATLAB Stormer_Verlet)."""
    q, p, h = 0.7, -0.3, 0.05
    p_half = p + 0.5 * h * (-q - 2 * ZETA * p)
    q_new = q + h * p_half
    p_new = p_half + 0.5 * h * (-q_new - 2 * ZETA * p_half)
    y_new = verlet_step(np.array([q, p]), h, acc=damped)
    assert abs(y_new[0] - q_new) < 1e-14
    assert abs(y_new[1] - p_new) < 1e-14


@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
@pytest.mark.parametrize("implicit", [False, True])
@pytest.mark.parametrize("problem, acc", [("harmonic", harmonic), ("damped", damped)])
def test_symplectic_euler_is_first_order(references, problem, acc, implicit):
    order = observed_order(symplectic_euler_step, references[problem], acc=acc, implicit=implicit)
    assert 0.9 < order < 1.1


@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
def test_implicit_symplectic_euler_matches_the_closed_form():
    """Damped oscillator: p_new = (p - h q) / (1 + 2 zeta h), q_new = q + h p_new."""
    q, p, h = 0.7, -0.3, 0.05
    y_new = symplectic_euler_step(np.array([q, p]), h, acc=damped, implicit=True)
    p_exact = (p - h * q) / (1 + 2 * ZETA * h)
    assert abs(y_new[1] - p_exact) < 1e-12
    assert abs(y_new[0] - (q + h * p_exact)) < 1e-12


def test_steps_work_in_three_dimensions():
    """The state is split in two halves: a 6-vector is (q1, q2, q3, p1, p2, p3)."""
    y0 = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    y = y0.copy()
    for _ in range(2000):
        y = verlet_step(y, 0.005, acc=harmonic)
    energy = lambda s: 0.5 * np.dot(s[3:], s[3:]) + 0.5 * np.dot(s[:3], s[:3])
    assert y.shape == (6,)
    assert abs(energy(y) - energy(y0)) < 1e-5