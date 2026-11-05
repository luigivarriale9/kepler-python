import numpy as np
from scipy.linalg import hilbert
from local_optimisation import (rosen, rosen_grad, rosen_hess, newton,
                                newton_armijo, sd_armijo, sd_exact, conjgrad, conjgrad_ls)

X0 = np.array([-1.2, 1.0])


def test_gradient_and_hessian_match_finite_differences():
    x = np.array([0.3, -0.7]); h = 1e-6
    g = np.array([(rosen(x + h * e) - rosen(x - h * e)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(rosen_grad(x), g, atol=1e-4)
    H = np.array([(rosen_grad(x + h * e) - rosen_grad(x - h * e)) / (2 * h) for e in np.eye(2)])
    assert np.allclose(rosen_hess(x), H, atol=1e-3)


def test_newton_finds_minimum():
    x, k = newton(rosen_grad, rosen_hess, X0)
    assert np.allclose(x, [1, 1], atol=1e-6) and k < 10


def test_newton_armijo_with_step_reset_is_fast():
    x, k = newton_armijo(rosen, rosen_grad, rosen_hess, X0, 1e-6, 1000, True)
    assert np.allclose(x, [1, 1], atol=1e-6) and k < 30


def test_newton_armijo_without_reset_is_much_slower():
    _, k_fedele = newton_armijo(rosen, rosen_grad, rosen_hess, X0, 1e-6, 1000, False)
    _, k_reset = newton_armijo(rosen, rosen_grad, rosen_hess, X0, 1e-6, 1000, True)
    assert k_fedele > 5 * k_reset


def test_steepest_descent_converges_slowly():
    x, k = sd_armijo(rosen, rosen_grad, X0, 1e-6, 100000)
    assert np.allclose(x, [1, 1], atol=1e-4) and k > 1000


def test_hilbert_cases_match_matlab_headers():
    A = hilbert(5); b = A @ np.ones(5); z = np.zeros(5)
    xs, ks = sd_exact(z, A, b, 1e-6, 10)
    xc, kc = conjgrad(z, A, b, 1e-6)
    assert ks == 11 and kc == 4
    assert np.allclose(xs, [0.9615, 1.1214, 1.0497, 0.9549, 0.8671], atol=1e-4)
    assert np.allclose(xc, [1.0000, 1.0005, 0.9976, 1.0036, 0.9983], atol=1e-4)


def test_conjgrad_ls_matches_lstsq_on_noisy_overdetermined_system():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((30, 5))
    b = A @ np.ones(5) + 0.1 * rng.standard_normal(30)
    x, k = conjgrad_ls(A, b, np.zeros(5), 1e-10)
    assert np.allclose(x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-6)
    assert k <= 5


def test_conjgrad_ls_underdetermined_gives_minimum_norm_solution():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 10))
    b = A @ np.ones(10)
    x, k = conjgrad_ls(A, b, np.zeros(10), 1e-10)
    assert np.allclose(A @ x, b, atol=1e-6)
    assert np.allclose(x, np.linalg.pinv(A) @ b, atol=1e-6)
    assert k <= 5
