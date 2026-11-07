"""Tests for gauss_jackson.py on the Kepler problem (mu = 1) and a J2 perturbation."""
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from gauss_jackson import coefficients, gauss_jackson

E = 0.7  # eccentricity; a = 1, period 2 pi, start at pericentre


def kepler_acc(t, y):
    return -y / np.linalg.norm(y) ** 3


def kepler_exact(t, e=E):
    """Exact planar Kepler orbit from pericentre (mu = a = 1) via Kepler's equation."""
    M = np.atleast_1d(t)
    En = M.copy()
    for _ in range(100):
        En -= (En - e * np.sin(En) - M) / (1 - e * np.cos(En))
    y = np.stack([np.cos(En) - e, np.sqrt(1 - e**2) * np.sin(En)], axis=1)
    v = np.stack([-np.sin(En), np.sqrt(1 - e**2) * np.cos(En)], axis=1) / (1 - e * np.cos(En))[:, None]
    return y, v


def run(h_div, order=8, T=2 * np.pi, e=E):
    n = h_div
    h = T / n
    y0, v0 = kepler_exact(0.0)
    t, Y, V = gauss_jackson(kepler_acc, 0.0, y0[0], v0[0], h, n, order=order)
    ye, ve = kepler_exact(t)
    return np.max(np.linalg.norm(Y - ye, axis=1)), np.max(np.linalg.norm(V - ve, axis=1)), t, Y, V


def test_coefficients_match_vallado_table():
    st, cw, ab, am = coefficients(6)
    assert st == [F(1), F(0), F(1, 12), F(1, 12), F(19, 240), F(3, 40)]
    assert cw == [F(1), F(-1), F(1, 12), F(0), F(-1, 240), F(-1, 240)]
    assert ab == [F(1), F(1, 2), F(5, 12), F(3, 8), F(251, 720), F(95, 288)]
    assert am == [F(1), F(-1, 2), F(-1, 12), F(-1, 24), F(-19, 720), F(-3, 160)]


def test_exact_for_polynomial():
    # y = t^7 has y'' of degree 5 -> annihilated by order >= 5 differences
    f = lambda t, y: np.array([42 * t**5])
    t, Y, V = gauss_jackson(f, 0.0, [0.0], [0.0], 0.1, 20, order=8)
    assert np.allclose(Y[:, 0], t**7, atol=1e-12)
    assert np.allclose(V[:, 0], 7 * t**6, atol=1e-12)


@pytest.mark.parametrize("order", [4, 6])
def test_convergence_order(order):
    # global position error is O(h^(order+2)); order 8 is not tested here because its
    # error reaches ~1e-11, the accuracy floor of the starting values and round-off
    n1, n2 = 800, 1600
    e1 = run(n1, order)[0]
    e2 = run(n2, order)[0]
    p = np.log2(e1 / e2)
    assert p > order + 1, f"observed order {p:.2f}"


def test_accuracy_and_velocity():
    ey, ev, *_ = run(2000)
    assert ey < 1e-8 and ev < 1e-8


def test_conserved_quantities_beat_rk4():
    """10 periods, 400 steps per period: energy / angular momentum drift vs classical RK4."""
    n = 4000
    T = 20 * np.pi
    _, _, t, Y, V = run(n, T=T)
    energy = lambda Y, V: 0.5 * (V**2).sum(1) - 1 / np.linalg.norm(Y, axis=1)
    L = lambda Y, V: Y[:, 0] * V[:, 1] - Y[:, 1] * V[:, 0]
    dE_gj = np.max(np.abs(energy(Y, V) - energy(Y, V)[0]))
    dL_gj = np.max(np.abs(L(Y, V) - L(Y, V)[0]))

    h = T / n
    g = lambda s: np.concatenate([s[2:], kepler_acc(0, s[:2])])
    s = np.concatenate([Y[0], V[0]])
    S = [s]
    for _ in range(n):
        k1 = g(s); k2 = g(s + h / 2 * k1); k3 = g(s + h / 2 * k2); k4 = g(s + h * k3)
        s = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        S.append(s)
    S = np.array(S)
    eR = energy(S[:, :2], S[:, 2:])
    dE_rk = np.max(np.abs(eR - eR[0]))
    print(f"max dE: Gauss-Jackson {dE_gj:.2e}, RK4 {dE_rk:.2e}; max dL: {dL_gj:.2e}")
    assert dE_gj < dE_rk / 10


def test_j2_nodal_precession():
    """Secular drift of the node under J2 vs analytic -1.5 J2 n (R/p)^2 cos i."""
    mu, R, J2 = 398600.4418, 6378.137, 1.08263e-3
    a, e, inc = 8000.0, 0.05, np.radians(50)
    p = a * (1 - e**2)
    rp = a * (1 - e)
    vp = np.sqrt(mu * (1 + e) / rp)
    y0 = np.array([rp, 0, 0])
    v0 = np.array([0, vp * np.cos(inc), vp * np.sin(inc)])

    def f(t, y):
        r = np.linalg.norm(y)
        k = 1.5 * J2 * mu * R**2 / r**5
        z2 = 5 * y[2] ** 2 / r**2
        return -mu * y / r**3 - k * y * np.array([1 - z2, 1 - z2, 3 - z2])

    n_mean = np.sqrt(mu / a**3)
    P = 2 * np.pi / n_mean
    n_orb = 20
    steps = n_orb * 600
    t, Y, V = gauss_jackson(f, 0.0, y0, v0, n_orb * P / steps, steps)
    hvec = np.cross(Y, V)
    node = np.unwrap(np.arctan2(hvec[:, 0], -hvec[:, 1]))  # RAAN
    rate = np.polyfit(t, node, 1)[0]
    expected = -1.5 * J2 * n_mean * (R / p) ** 2 * np.cos(inc)
    assert rate == pytest.approx(expected, rel=0.02)