import numpy as np
from scipy.integrate import solve_ivp
from sampled_vs_linear_volume import (initial_state, rhs_state, sqrt_det_cov,
                                      propagate_cloud, run, n)

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
