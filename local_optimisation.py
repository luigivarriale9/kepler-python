"""Metodi di ottimizzazione locale (traduzione Python dei Lab 7-9 di Calcolo Scientifico).

Originale MATLAB: elaborato consegnato in coppia (Mattinale, Varriale).
Traduzione fedele, con un'opzione `reset_step` per vedere l'effetto del passo.
"""
import numpy as np


def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosen_grad(x):
    return np.array([-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                     200 * (x[1] - x[0] ** 2)])


def rosen_hess(x):
    return np.array([[2 - 400 * x[1] + 1200 * x[0] ** 2, -400 * x[0]],
                     [-400 * x[0], 200.0]])


def _backtrack(f, x, grad, pk, lam, alpha=1e-4, rho=0.5):
    """Riduce lam finche' vale la condizione di Armijo."""
    xnew = x + lam * pk
    while f(x) + alpha * lam * (grad @ pk) <= f(xnew):
        lam *= rho
        xnew = x + lam * pk
    return xnew, lam


def sd_armijo(f, df, x0, tol=1e-6, maxit=100, reset_step=False):
    """Discesa del gradiente con backtracking. Ritorna (x, k)."""
    x = np.asarray(x0, dtype=float)
    k, lam, ngrad = 0, 1.0, tol + 1
    while k <= maxit and ngrad > tol:
        if reset_step:
            lam = 1.0
        g = df(x)
        x, lam = _backtrack(f, x, g, -g, lam)
        ngrad = np.linalg.norm(df(x))
        k += 1
    return x, k


def newton(df, hf, x0, tol=1e-6, maxit=1000):
    """Newton puro per la minimizzazione. Ritorna (x, k)."""
    x = np.asarray(x0, dtype=float)
    k, ngrad = 0, tol + 1
    while k <= maxit and ngrad > tol:
        k += 1
        x = x - np.linalg.solve(hf(x), df(x))
        ngrad = np.linalg.norm(df(x))
    return x, k


def newton_armijo(f, df, hf, x0, tol=1e-6, maxit=100, reset_step=False):
    """Newton con backtracking. Ritorna (x, k)."""
    x = np.asarray(x0, dtype=float)
    k, lam, ngrad = 0, 1.0, tol + 1
    while k <= maxit and ngrad > tol:
        if reset_step:
            lam = 1.0
        g = df(x)
        pk = -np.linalg.solve(hf(x), g)
        x, lam = _backtrack(f, x, g, pk, lam)
        ngrad = np.linalg.norm(df(x))
        k += 1
    return x, k


def sd_exact(x0, A, b, tau=1e-6, maxit=10):
    """Discesa del gradiente con ricerca esatta per Ax=b (A simmetrica def. pos.)."""
    xk = np.asarray(x0, dtype=float)
    gk = A @ xk - b
    pk = -gk
    k = 0
    while tau <= np.linalg.norm(A @ xk - b) and k <= maxit:
        lam = -(gk @ pk) / (pk @ A @ pk)
        xk = xk + lam * pk
        gk = A @ xk - b
        pk = -gk
        k += 1
    return xk, k


def conjgrad(x0, A, b, tau=1e-6):
    """Gradiente coniugato per Ax=b, A simmetrica definita positiva."""
    n = A.shape[0]
    xk = np.asarray(x0, dtype=float)
    gk = A @ xk - b
    pk = -gk
    k = 0
    while tau <= np.linalg.norm(A @ xk - b) and k <= n:
        gs = gk @ gk
        ps = A @ pk
        lam = gs / (pk @ ps)
        xk = xk + lam * pk
        gnew = gk + lam * ps
        beta = (gnew @ gnew) / gs
        gk = gnew
        pk = -gk + beta * pk
        k += 1
    return xk, k


if __name__ == "__main__":
    x0 = np.array([-1.2, 1.0])
    print("Rosenbrock from (-1.2, 1), tol 1e-6")
    print(f"{'method':38s}{'iterations':>11s}   final f")
    rows = [
        ("Newton", newton(rosen_grad, rosen_hess, x0)),
        ("Newton + Armijo (step never reset)", newton_armijo(rosen, rosen_grad, rosen_hess, x0, 1e-6, 1000)),
        ("Newton + Armijo (step reset to 1)", newton_armijo(rosen, rosen_grad, rosen_hess, x0, 1e-6, 1000, True)),
        ("Steepest descent + Armijo", sd_armijo(rosen, rosen_grad, x0, 1e-6, 100000)),
    ]
    for name, (x, k) in rows:
        print(f"{name:38s}{k:11d}   {rosen(x):.2e}")
