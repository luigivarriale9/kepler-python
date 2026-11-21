import os

import numpy as np
import pytest

import three_body_comparison as T
from pcr3bp import jacobi


def test_default_initial_state_has_the_jacobi_constant_of_the_matlab_test_case():
    assert jacobi(T.Y0, T.MU) == pytest.approx(3.1624, abs=1e-4)


def test_tight_ode45_conserves_the_jacobi_constant():
    t, Y = T.solve("ODE45 tight", T.Y0)
    C = jacobi(Y, T.MU)
    assert t[-1] == pytest.approx(T.T_END)
    assert np.max(np.abs(C - C[0])) < 1e-9

@pytest.mark.filterwarnings("ignore:The iteration is not making good progress")
def test_main_prints_the_report_and_saves_the_figures(capsys, tmp_path):
    T.main(["--methods", "Implicit midpoint", "Explicit Euler", "--steps-scale", "0.01",
            "--c", "3.1", "--outdir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Jacobi constant used: C = 3.100000" in out
    assert os.path.exists(tmp_path / "cr3bp_trajectories.png")
    assert os.path.exists(tmp_path / "cr3bp_jacobi_drift.png")