import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import arenstorf_comparison as A
from pcr3bp import Y0_ARENSTORF, hamiltonian, rhs, to_canonical


def _states():
    rng = np.random.default_rng(0)
    return np.column_stack([rng.uniform(-1.2, 1.2, 6), rng.uniform(0.1, 1.0, 6),
                            rng.normal(size=6), rng.normal(size=6)])


def _observed_order(name, ns, t_end):
    ref = solve_ivp(rhs, (0, t_end), Y0_ARENSTORF, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]
    ns = np.array(ns)
    errors = [np.linalg.norm(A.solve(name, int(n), t_end=t_end)[-1] - ref) for n in ns]
    return np.polyfit(np.log(t_end / ns), np.log(errors), 1)[0]


def test_energy_matches_the_canonical_hamiltonian():
    S = _states()
    H = np.array([hamiltonian(to_canonical(s)) for s in S])
    assert np.max(np.abs(A.energy(S) - H)) < 1e-13

def test_reference_closes_the_orbit_and_conserves_the_energy():
    Y = A.solve("ODE45", 2000)
    E = A.energy(Y)
    assert np.linalg.norm(Y[-1] - Y0_ARENSTORF) < 1e-3
    assert np.max(np.abs(E - E[0])) < 1e-6


@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
def test_implicit_midpoint_is_second_order_on_a_short_arc():
    assert 1.8 < _observed_order("Implicit midpoint", [1000, 2000, 4000], t_end=1.0) < 2.2


def test_stormer_verlet_is_first_order_with_the_coriolis_term():
    """Fine steps are needed: the error term of order h^2 dominates for n of a few thousand."""
    assert 0.9 < _observed_order("Stormer-Verlet", [32000, 128000], t_end=1.0) < 1.3


def test_compare_returns_the_requested_solutions():
    solutions = A.compare(200, ["Explicit Euler", "Stormer-Verlet"])
    assert list(solutions) == ["ODE45", "Explicit Euler", "Stormer-Verlet"]
    for Y in solutions.values():
        assert Y.shape == (201, 4)
        assert np.allclose(Y[0], Y0_ARENSTORF)


def test_main_prints_the_report_and_saves_the_figures(capsys, tmp_path):
    A.main(["--n", "200", "--methods", "Explicit Euler", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Maximum energy error" in out and "Distance from y0 after one period" in out
    assert os.path.exists(tmp_path / "arenstorf_orbits.png")
    assert os.path.exists(tmp_path / "arenstorf_energy.png")