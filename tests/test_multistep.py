import numpy as np
import pytest
from kepler import initial_state, rhs, reference
from multistep import adams_bashforth, adams_moulton, abm4_pece


def _order(errs, ns, t_end=1.0):
    return np.polyfit(np.log(t_end / np.asarray(ns)), np.log(errs), 1)[0]


def _scalar_errors(method, ns):
    f = lambda t, y: -y
    exact = np.exp(-1.0)
    return [abs(method(f, 0.0, 1.0, [1.0], n)[1][-1, 0] - exact) for n in ns]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_adams_bashforth_order(k):
    ns = [50, 100, 200]
    errs = _scalar_errors(lambda f, a, b, y0, n: adams_bashforth(f, a, b, y0, n, k), ns)
    assert abs(_order(errs, ns) - k) < 0.2


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_adams_moulton_order(k):
    ns = [10, 20, 40] if k == 4 else [50, 100, 200]
    errs = _scalar_errors(lambda f, a, b, y0, n: adams_moulton(f, a, b, y0, n, k), ns)
    assert abs(_order(errs, ns) - (k + 1)) < 0.3


def test_pece_matches_known_error():
    # y' = 1 + y^2, y(0) = 1: exact solution tan(t + pi/4)
    f = lambda t, y: 1 + y ** 2
    t, y = abm4_pece(f, 0.0, 0.75, [1.0], 1000)
    err = abs(y[-1, 0] - np.tan(0.75 + np.pi / 4))
    assert err < 1e-6


def test_pece_on_kepler_beats_ab4():
    y0, t_end, n = initial_state(0.7), 3.0, 500
    y_ref = reference(y0, t_end)
    e_pece = np.linalg.norm(abm4_pece(rhs, 0.0, t_end, y0, n)[1][-1] - y_ref)
    e_ab4 = np.linalg.norm(adams_bashforth(rhs, 0.0, t_end, y0, n, 4)[1][-1] - y_ref)
    assert e_pece < e_ab4
