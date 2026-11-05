"""Linear multistep methods of Adams type: Adams-Bashforth (AB), Adams-Moulton (AM)
and the predictor-corrector scheme ABM4 in PECE form (AB4 predictor, AM4 corrector).

Conventions
-----------
All methods solve y' = f(t, y), y(t0) = y0 on [t0, t_end] with n equal steps.
The coefficient tables below list the weights of f_{n-k+1}, ..., f_n (AB) or
f_{n-k+1}, ..., f_{n+1} (AM), in this order, divided by a common denominator.

    AB_k (k = 1..4): order k
    AM_k (k = 1..4): order k + 1   (AM1 is the trapezoidal rule)

Starting values are computed with DOP853 at tight tolerances, so that the error
of the multistep method is not polluted by the starting procedure.
Implicit methods are solved by fixed-point iteration.
"""
import numpy as np
from scipy.integrate import solve_ivp

# (weights, denominator)
AB_COEFFS = {
    1: ([1.0], 1.0),
    2: ([-1.0, 3.0], 2.0),
    3: ([5.0, -16.0, 23.0], 12.0),
    4: ([-9.0, 37.0, -59.0, 55.0], 24.0),
}
AM_COEFFS = {
    1: ([1.0, 1.0], 2.0),
    2: ([-1.0, 8.0, 5.0], 12.0),
    3: ([1.0, -5.0, 19.0, 9.0], 24.0),
    4: ([-19.0, 106.0, -264.0, 646.0, 251.0], 720.0),
}


def _starting_values(f, t0, y0, h, k):
    """y at t0, t0+h, ..., t0+(k-1)h computed with DOP853."""
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    ys = [y0]
    if k > 1:
        ts = t0 + h * np.arange(1, k)
        sol = solve_ivp(f, (t0, ts[-1]), y0, method="DOP853",
                        t_eval=ts, rtol=1e-13, atol=1e-13)
        ys.extend(sol.y.T)
    return ys


def _setup(f, t0, t_end, y0, n, k):
    h = (t_end - t0) / n
    t = t0 + h * np.arange(n + 1)
    y = np.zeros((n + 1, np.size(y0)))
    F = np.zeros_like(y)
    for j, yj in enumerate(_starting_values(f, t0, y0, h, k)):
        y[j] = yj
        F[j] = f(t[j], y[j])
    return h, t, y, F


def adams_bashforth(f, t0, t_end, y0, n, k):
    """Adams-Bashforth method with k steps (order k). Returns (t, y)."""
    w, den = AB_COEFFS[k]
    h, t, y, F = _setup(f, t0, t_end, y0, n, k)
    for i in range(k - 1, n):
        y[i + 1] = y[i] + h / den * sum(w[j] * F[i - k + 1 + j] for j in range(k))
        F[i + 1] = f(t[i + 1], y[i + 1])
    return t, y


def adams_moulton(f, t0, t_end, y0, n, k, tol=1e-12, max_iter=100):
    """Adams-Moulton method with k steps (order k + 1), implicit equation solved
    by fixed-point iteration. Returns (t, y)."""
    w, den = AM_COEFFS[k]
    h, t, y, F = _setup(f, t0, t_end, y0, n, k)
    for i in range(k - 1, n):
        known = y[i] + h / den * sum(w[j] * F[i - k + 1 + j] for j in range(k))
        z = y[i] + h * F[i]                      # initial guess: explicit Euler
        for _ in range(max_iter):
            z_new = known + h * w[k] / den * f(t[i + 1], z)
            done = np.linalg.norm(z_new - z) < tol
            z = z_new
            if done:
                break
        else:
            raise RuntimeError("fixed-point iteration did not converge; reduce h")
        y[i + 1] = z
        F[i + 1] = f(t[i + 1], z)
    return t, y


def abm4_pece(f, t0, t_end, y0, n):
    """Predictor-corrector ABM4: P = AB4, E, C = AM4, E (PECE). Returns (t, y)."""
    wp, dp = AB_COEFFS[4]
    wc, dc = AM_COEFFS[4]
    k = 4
    h, t, y, F = _setup(f, t0, t_end, y0, n, k)
    for i in range(k - 1, n):
        # P: predictor with AB4
        yp = y[i] + h / dp * sum(wp[j] * F[i - 3 + j] for j in range(4))
        # E: evaluate f at the predicted value
        fp = f(t[i + 1], yp)
        # C: corrector with AM4 (weights of f_{n-3}, ..., f_n, f_{n+1}); f_{n+1} is fp
        y[i + 1] = y[i] + h / dc * (sum(wc[j] * F[i - 3 + j] for j in range(4)) + wc[4] * fp)
        # E: evaluate f at the corrected value
        F[i + 1] = f(t[i + 1], y[i + 1])
    return t, y
