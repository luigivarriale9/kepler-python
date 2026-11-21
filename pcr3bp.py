"""Planar circular restricted three-body problem (PCR3BP), rotating frame.

Normalised units: total mass 1, distance between the primaries 1, angular velocity 1.
The primary of mass 1 - MU sits at (-MU, 0), the one of mass MU at (1 - MU, 0).
State s = (x, y, vx, vy). Canonical variables: q = (x, y), p = (vx - y, vy + x).
Test case: periodic Arenstorf orbit (Earth-Moon, MU = 0.012277471).
"""
import numpy as np
from scipy.optimize import brentq

MU = 0.012277471
T_ARENSTORF = 17.0652165601579625588917206249
Y0_ARENSTORF = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])


def _distances(x, y, mu=MU):
    r1 = np.hypot(x + mu, y)          # distance from the primary of mass 1 - mu
    r2 = np.hypot(x - 1 + mu, y)      # distance from the primary of mass mu
    return r1, r2


def rhs(t, s, mu=MU):
    """Equations of motion in the variables (x, y, vx, vy)."""
    x, y, vx, vy = s
    r1, r2 = _distances(x, y, mu)
    ax = x + 2 * vy - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3
    ay = y - 2 * vx - (1 - mu) * y / r1**3 - mu * y / r2**3
    return np.array([vx, vy, ax, ay])


def acceleration(q, v, mu=MU):
    """Acceleration (ax, ay) as a function of position q and velocity v.

    It depends on v through the Coriolis term, so the partitioned methods of kepler.py
    (Stormer-Verlet, symplectic Euler) need this function as `acc`."""
    return rhs(0.0, np.concatenate([q, v]), mu)[2:]


def to_canonical(s):
    """(x, y, vx, vy) -> (x, y, px, py)."""
    x, y, vx, vy = s
    return np.array([x, y, vx - y, vy + x])


def from_canonical(z):
    """(x, y, px, py) -> (x, y, vx, vy)."""
    x, y, px, py = z
    return np.array([x, y, px + y, py - x])


def hamiltonian(z, mu=MU):
    """H(q, p) = (px^2 + py^2)/2 + y px - x py - (1-mu)/r1 - mu/r2."""
    x, y, px, py = z
    r1, r2 = _distances(x, y, mu)
    return 0.5 * (px**2 + py**2) + y * px - x * py - (1 - mu) / r1 - mu / r2


def zero_velocity_jacobi(x, y, mu=MU):
    """Jacobi constant of a state at rest at (x, y): x^2 + y^2 + 2(1-mu)/r1 + 2 mu/r2.

    A trajectory with constant C can only visit the points where this is >= C, so the
    forbidden region is {(x, y): zero_velocity_jacobi(x, y) < C}."""
    r1, r2 = _distances(x, y, mu)
    return x**2 + y**2 + 2 * (1 - mu) / r1 + 2 * mu / r2


def jacobi(s, mu=MU):
    """Jacobi constant C = -2H = x^2 + y^2 + 2(1-mu)/r1 + 2 mu/r2 - v^2.

    s is one state (4,) or an array of states (N, 4)."""
    s = np.asarray(s, dtype=float)
    return zero_velocity_jacobi(s[..., 0], s[..., 1], mu) - s[..., 2]**2 - s[..., 3]**2


def rescale_to_jacobi(s, c, mu=MU):
    """Same position and direction of the velocity as s, magnitude chosen so that C = c."""
    v2 = zero_velocity_jacobi(s[0], s[1], mu) - c
    if v2 < 0:
        raise ValueError(f"C too high: no real velocity at this point (C <= {zero_velocity_jacobi(s[0], s[1], mu):.4f})")
    return np.concatenate([s[:2], s[2:] * np.sqrt(v2) / np.linalg.norm(s[2:])])


def lagrange_points(mu=MU):
    """The five equilibrium points L1..L5 as an array (5, 2).

    L1, L2, L3 are the zeros of dOmega/dx on the x axis; L4 and L5 form equilateral
    triangles with the primaries."""
    def d_omega(x):
        return x - (1 - mu) * (x + mu) / abs(x + mu)**3 - mu * (x - 1 + mu) / abs(x - 1 + mu)**3
    xl1 = brentq(d_omega, -mu + 0.05, 1 - mu - 0.05)
    xl2 = brentq(d_omega, 1 - mu + 0.05, 2.0)
    xl3 = brentq(d_omega, -2.0, -mu - 0.05)
    return np.array([[xl1, 0.0], [xl2, 0.0], [xl3, 0.0],
                     [0.5 - mu, np.sqrt(3) / 2], [0.5 - mu, -np.sqrt(3) / 2]])