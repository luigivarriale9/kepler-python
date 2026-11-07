"""Gauss-Jackson (Stormer-Cowell) multistep integrator for y'' = f(t, y).

Fixed step h, backward-difference form, PECE:
    predictor (Stormer):  y_{n+1} = 2 y_n - y_{n-1} + h^2 * sum_j gamma_j * nabla^j f_n
    corrector (Cowell):   y_{n+1} = 2 y_n - y_{n-1} + h^2 * sum_j delta_j * nabla^j f_{n+1}
    velocity (Adams):     v_{n+1} = v_n + h * sum_j (AB|AM)_j * nabla^j f_(n|n+1)
Order k means differences up to nabla^k are kept (local error O(h^(k+3)), global error O(h^(k+2))
in position, until the accuracy of the starting values and round-off is reached).
Coefficients come from generating functions in exact rational arithmetic
(Vallado, Fundamentals of Astrodynamics and Applications, Sec. 7.5):
    Cowell  (t / ln(1-t))^2,   Stormer = Cowell / (1-t),
    Adams-Moulton t / (-ln(1-t)),   Adams-Bashforth = AM / (1-t).
This is the difference form, not the summed-ordinate form of Berry & Healy.
"""
from fractions import Fraction as F

import numpy as np
from scipy.integrate import solve_ivp


def _mul(a, b, n):
    r = [F(0)] * n
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if i + j < n:
                r[i + j] += x * y
    return r


def _inv(a, n):
    r = [F(0)] * n
    r[0] = 1 / a[0]
    for m in range(1, n):
        r[m] = -sum(a[k] * r[m - k] for k in range(1, m + 1)) / a[0]
    return r


def coefficients(n=10):
    """Return (stormer, cowell, adams_bashforth, adams_moulton) as lists of Fractions."""
    q = _inv([F(1, k + 1) for k in range(n)], n)   # t / (-ln(1-t))
    ones = [F(1)] * n                              # 1 / (1-t)
    cowell = _mul(q, q, n)
    return _mul(cowell, ones, n), cowell, _mul(q, ones, n), q


def _differences(fhist, order):
    """nabla^j f_last for j = 0..order from the last order+1 accelerations."""
    d = [fhist[-1]]
    cur = list(fhist[-order - 1:])
    for _ in range(order):
        cur = [cur[i + 1] - cur[i] for i in range(len(cur) - 1)]
        d.append(cur[-1])
    return d


def gauss_jackson(f, t0, y0, v0, h, n_steps, order=8, n_corr=1, start=None):
    """Integrate y'' = f(t, y) with fixed step h.

    f(t, y) -> acceleration (array like y). Returns t, Y, V with n_steps+1 rows.
    start: optional (Y, V) arrays with the first `order+1` states at t0 + k h;
    by default they are computed with DOP853 (rtol 1e-13).
    """
    y0 = np.atleast_1d(np.asarray(y0, float))
    v0 = np.atleast_1d(np.asarray(v0, float))
    d = y0.size
    ns = order + 1
    if n_steps < ns:
        raise ValueError("n_steps must be >= order + 1")
    g, c, ab, am = [np.array([float(x) for x in cf[:order + 1]]) for cf in coefficients(order + 1)]

    t = t0 + h * np.arange(n_steps + 1)
    Y = np.zeros((n_steps + 1, d))
    V = np.zeros((n_steps + 1, d))
    if start is None:
        sol = solve_ivp(lambda s, u: np.concatenate([u[d:], f(s, u[:d])]), (t0, t[ns - 1]),
                        np.concatenate([y0, v0]), method="DOP853", t_eval=t[:ns],
                        rtol=1e-13, atol=1e-14)
        Y[:ns], V[:ns] = sol.y[:d].T, sol.y[d:].T
    else:
        Y[:ns], V[:ns] = start[0][:ns], start[1][:ns]
    F_ = [f(t[i], Y[i]) for i in range(ns)]

    for n in range(ns - 1, n_steps):
        dn = _differences(F_, order)
        yp = 2 * Y[n] - Y[n - 1] + h**2 * sum(g[j] * dn[j] for j in range(order + 1))
        vp = V[n] + h * sum(ab[j] * dn[j] for j in range(order + 1))
        for _ in range(n_corr):
            fnew = f(t[n + 1], yp)
            dc = _differences(F_ + [fnew], order)
            yp = 2 * Y[n] - Y[n - 1] + h**2 * sum(c[j] * dc[j] for j in range(order + 1))
            vp = V[n] + h * sum(am[j] * dc[j] for j in range(order + 1))
        Y[n + 1], V[n + 1] = yp, vp
        F_.append(f(t[n + 1], yp))
        if len(F_) > order + 2:
            F_.pop(0)
    return t, Y, V

if __name__ == "__main__":
    # Demo: Kepler problem, e = 0.7, one period, against the exact starting point.
    e = 0.7
    acc = lambda t, y: -y / np.linalg.norm(y) ** 3
    y0 = np.array([1 - e, 0.0])
    v0 = np.array([0.0, np.sqrt((1 + e) / (1 - e))])
    print("Kepler problem, e = 0.7, one period, order 8")
    print(f"{'steps':>7} {'closure error':>15} {'max energy drift':>18}")
    for n in (250, 500, 1000, 2000):
        t, Y, V = gauss_jackson(acc, 0.0, y0, v0, 2 * np.pi / n, n, order=8)
        En = 0.5 * (V**2).sum(1) - 1 / np.linalg.norm(Y, axis=1)
        print(f"{n:7d} {np.linalg.norm(Y[-1] - y0):15.2e} {np.max(np.abs(En - En[0])):18.2e}")
