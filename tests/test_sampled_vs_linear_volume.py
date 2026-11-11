import numpy as np
from scipy.integrate import solve_ivp
import pytest
from sampled_vs_linear_volume import (initial_state, rhs_state, sqrt_det_cov,
                                      propagate_cloud, run, n,
                                      cartesian_to_delaunay, sqrt_det_cov_delaunay,
                                      wrap)

E = 0.7


def test_ratio_at_t0_is_one():
    # The initial cloud has covariance P0: the sampled volume must match
    # sqrt(det P0) up to sampling noise.
    sigma, N = 1e-6, 2000
    rng = np.random.default_rng(0)
    P0 = np.diag([sigma**2] * (2 * n))
    X0 = rng.multivariate_normal(initial_state(E), P0, size=N)
    ratio = sqrt_det_cov(X0) / np.sqrt(np.linalg.det(P0))
    assert abs(ratio - 1) < 0.05


def test_kepler_energy_and_period():
    x0 = initial_state(E)
    t_eval = np.linspace(0, 2 * np.pi, 200)
    sol = solve_ivp(rhs_state, (0, 2 * np.pi), x0, method="DOP853",
                    t_eval=t_eval, rtol=1e-12, atol=1e-12)
    q, v = sol.y[:n, :], sol.y[n:, :]
    energy = 0.5 * np.sum(v**2, axis=0) - 1.0 / np.linalg.norm(q, axis=0)
    assert np.all(np.abs(energy + 0.5) < 1e-9)          # E = -1/2
    assert np.max(np.abs(sol.y[:, -1] - x0)) < 1e-8     # period 2*pi (a = 1)


def test_cloud_shape():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal(initial_state(E), 1e-12 * np.eye(4), size=20)
    traj = propagate_cloud(X, np.arange(1, 4) * 2 * np.pi, n_proc=2)
    assert traj.shape == (3, 20, 4)


def test_linear_limit():
    # For a very small cloud the propagation is linear: ratio close to 1.
    r = run(1e-10, N=1000, periods=1, n_proc=2)[0]
    assert abs(r - 1) < 0.15


def test_nonlinear_growth_with_sigma():
    # Sampled volume grows with the size of the cloud after one period.
    r_small = run(1e-9, N=500, periods=1, n_proc=2)[0]
    r_large = run(1e-7, N=500, periods=1, n_proc=2)[0]
    assert r_large > 10 * r_small


def test_verlet_converges_to_dop853():
    # The result must not depend on the integrator once the Verlet step is
    # small enough (1600 steps per period).
    sigma, N = 1e-8, 500
    ref = run(sigma, N, periods=1, n_proc=2)[0]
    ver = run(sigma, N, periods=1, method="verlet", steps_per_period=1600)[0]
    assert abs(ver - ref) / ref < 0.02


# ---------- Delaunay coordinates ----------
def test_delaunay_initial_state():
    # Periapsis, a = 1: (l, g, L, G) = (0, 0, 1, sqrt(1 - e^2))
    D = cartesian_to_delaunay(np.array([initial_state(E)]))[0]
    expected = np.array([0.0, 0.0, 1.0, np.sqrt(1 - E**2)])
    assert np.max(np.abs(D - expected)) < 1e-12


def test_delaunay_constants_along_orbit():
    # Kepler flow: g, L, G constant and l(t) = t (a = 1, mu = 1).
    t_eval = np.linspace(0.1, 12.0, 40)
    sol = solve_ivp(rhs_state, (0, t_eval[-1]), initial_state(E), method="DOP853",
                    t_eval=t_eval, rtol=1e-12, atol=1e-12)
    D = cartesian_to_delaunay(sol.y.T)
    assert np.max(np.abs(wrap(D[:, 0] - t_eval))) < 1e-9
    assert np.max(D[:, 1:].std(axis=0)) < 1e-9


def _jacobian_det(x, h=1e-6):
    J = np.zeros((4, 4))
    for k in range(4):
        dx = np.zeros(4)
        dx[k] = h
        d = (cartesian_to_delaunay((x + dx)[None])[0]
             - cartesian_to_delaunay((x - dx)[None])[0])
        d[:2] = wrap(d[:2])
        J[:, k] = d / (2 * h)
    return np.linalg.det(J)


def test_delaunay_map_is_canonical():
    # A canonical map has |det J| = 1: the phase-space volume is preserved.
    t_eval = np.array([0.5, 2.0, 4.0, 6.0])
    sol = solve_ivp(rhs_state, (0, t_eval[-1]), initial_state(E), method="DOP853",
                    t_eval=t_eval, rtol=1e-12, atol=1e-12)
    for k in range(len(t_eval)):
        assert abs(abs(_jacobian_det(sol.y[:, k])) - 1) < 1e-6


def test_volume_equal_at_t0_in_both_coordinates():
    sigma, N = 1e-6, 2000
    rng = np.random.default_rng(1)
    X0 = rng.multivariate_normal(initial_state(E), np.diag([sigma**2] * 4), size=N)
    ratio = sqrt_det_cov(X0) / sqrt_det_cov_delaunay(cartesian_to_delaunay(X0))
    assert abs(ratio - 1) < 1e-3


def test_delaunay_volume_stays_linear_while_cartesian_does_not():
    # Same cloud, one period, sigma = 1e-6: Cartesian ratio is huge,
    # Delaunay ratio stays close to 1.
    cart = run(1e-6, N=1000, periods=1, n_proc=2)[0]
    dela = run(1e-6, N=1000, periods=1, n_proc=2, coords="delaunay")[0]
    assert cart > 100
    assert abs(dela - 1) < 0.05


def test_verlet_delaunay_close_to_one():
    r = run(1e-8, N=1000, periods=1, method="verlet", steps_per_period=1600,
            coords="delaunay")[0]
    assert abs(r - 1) < 0.05


def test_unknown_options_raise():
    with pytest.raises(ValueError):
        run(1e-8, N=50, periods=1, n_proc=1, coords="polar")
    with pytest.raises(ValueError):
        run(1e-8, N=50, periods=1, n_proc=1, reference="other")
