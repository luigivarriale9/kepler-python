"""Planar circular restricted three-body problem (PCR3BP), rotating frame.

Normalised units: total mass 1, distance between the primaries 1, angular velocity 1.
The primary of mass 1 - MU sits at (-MU, 0), the one of mass MU at (1 - MU, 0).
State s = (x, y, vx, vy). Canonical variables: q = (x, y), p = (vx - y, vy + x).
Test case: periodic Arenstorf orbit (Earth-Moon, MU = 0.012277471).
"""
import numpy as np

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


def jacobi(s, mu=MU):
    """Jacobi constant C = -2H (in terms of the state s)."""
    return -2 * hamiltonian(to_canonical(s), mu)
