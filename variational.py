"""Variational equations and state-transition matrix (STM) for the planar Kepler problem.

The state is x = (q, p) = (x, y, vx, vy), mu = 1.  The STM Phi = dx(t)/dx0 satisfies
    Phi' = A(x) Phi,   A = [[0, I], [G, 0]],   G = da/dq = -I/r^3 + 3 q q^T / r^5,   Phi(0) = I.
For a Hamiltonian system the exact flow is symplectic: Phi^T J Phi = J, det Phi = 1,
with J = [[0, I], [-I, 0]].

Each numerical method is a map x_{n+1} = Psi_h(x_n).  Its Jacobian M_n = dPsi_h/dx_n is
propagated together with the state, so Phi_N = M_N ... M_1 is the exact STM *of the scheme*
(not a finite-difference estimate).  For Runge-Kutta and partitioned methods applying the
method to the augmented system (x, Phi) gives exactly this Jacobian; here it is written
out by hand for the symplectic schemes.  A step acts on w = (x, vec(Phi)), 4 + 16 numbers.
Reference: Vallado, Fundamentals of Astrodynamics and Applications, Sec. 9.5.3;
Hairer, Lubich, Wanner, Geometric Numerical Integration, Ch. VI.
"""
import numpy as np
from scipy.integrate import solve_ivp

from kepler import acceleration, rhs

J = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
I2 = np.eye(2)
O2 = np.zeros((2, 2))


def gravity_gradient(q):
    """G = da/dq for a = -q / |q|^3 (mu = 1)."""
    r = np.linalg.norm(q)
    return -np.eye(2) / r**3 + 3.0 * np.outer(q, q) / r**5


def jacobian(x):
    """A = df/dx for f(x) = (p, a(q))."""
    return np.block([[O2, I2], [gravity_gradient(x[:2]), O2]])


def pack(x, Phi):
    return np.concatenate([x, Phi.ravel()])


def unpack(w):
    return w[:4], w[4:].reshape(4, 4)


def initial_augmented(x0):
    return pack(x0, np.eye(4))


# ---- one step of each method, acting on w = (x, vec Phi) -------------------------------

def euler_step_aug(w, h):
    x, Phi = unpack(w)
    M = np.eye(4) + h * jacobian(x)
    return pack(x + h * rhs(0.0, x), M @ Phi)


def symplectic_euler_step_aug(w, h):
    x, Phi = unpack(w)
    q, p = x[:2], x[2:]
    G = gravity_gradient(q)
    p_new = p + h * acceleration(q)
    q_new = q + h * p_new
    M = np.block([[I2 + h**2 * G, h * I2], [h * G, I2]])
    return pack(np.concatenate([q_new, p_new]), M @ Phi)


def verlet_step_aug(w, h):
    x, Phi = unpack(w)
    q, p = x[:2], x[2:]
    p_half = p + 0.5 * h * acceleration(q)
    q_new = q + h * p_half
    p_new = p_half + 0.5 * h * acceleration(q_new)
    K1 = np.block([[I2, O2], [0.5 * h * gravity_gradient(q), I2]])
    D = np.block([[I2, h * I2], [O2, I2]])
    K2 = np.block([[I2, O2], [0.5 * h * gravity_gradient(q_new), I2]])
    return pack(np.concatenate([q_new, p_new]), (K2 @ D @ K1) @ Phi)


def midpoint_step_aug(w, h, tol=1e-15, maxit=50):
    """Implicit midpoint; the stage equation is solved with Newton to machine precision,
    so that the symplecticity of the map is not limited by the solver tolerance."""
    x, Phi = unpack(w)
    z = x + h * rhs(0.0, x)
    for _ in range(maxit):
        xm = 0.5 * (x + z)
        F = z - x - h * rhs(0.0, xm)
        dz = np.linalg.solve(np.eye(4) - 0.5 * h * jacobian(xm), -F)
        z = z + dz
        if np.linalg.norm(dz) < tol:
            break
    A = jacobian(0.5 * (x + z))
    M = np.linalg.solve(np.eye(4) - 0.5 * h * A, np.eye(4) + 0.5 * h * A)
    return pack(z, M @ Phi)


def rk4_step_aug(w, h):
    """Classical RK4 applied to the augmented system (non-symplectic reference)."""
    def f(w):
        x, Phi = unpack(w)
        return pack(rhs(0.0, x), jacobian(x) @ Phi)
    k1 = f(w); k2 = f(w + 0.5 * h * k1); k3 = f(w + 0.5 * h * k2); k4 = f(w + h * k3)
    return w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


METHODS_AUG = {
    "Explicit Euler": euler_step_aug,
    "Symplectic Euler": symplectic_euler_step_aug,
    "Implicit midpoint": midpoint_step_aug,
    "Störmer–Verlet": verlet_step_aug,
    "RK4": rk4_step_aug,
}


def reference_stm(x0, t_end):
    """State and STM from DOP853 at tight tolerances (variational equations integrated
    together with the state)."""
    def f(t, w):
        x, Phi = unpack(w)
        return pack(rhs(t, x), jacobian(x) @ Phi)
    sol = solve_ivp(f, (0.0, t_end), initial_augmented(x0), method="DOP853",
                    rtol=1e-13, atol=1e-14)
    return unpack(sol.y[:, -1])


def symplecticity_error(Phi):
    """|| Phi^T J Phi - J ||_F."""
    return np.linalg.norm(Phi.T @ J @ Phi - J)


def propagate(step, x0, t_end, n_steps):
    """Return the time grid and the sequence of augmented states."""
    h = t_end / n_steps
    W = np.zeros((n_steps + 1, 20))
    W[0] = initial_augmented(x0)
    for k in range(n_steps):
        W[k + 1] = step(W[k], h)
    return h * np.arange(n_steps + 1), W
