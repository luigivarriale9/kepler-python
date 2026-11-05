import numpy as np
from compare_local_global import local_runs, global_run, TARGET


def test_differential_evolution_reaches_hohmann():
    f, _ = global_run(seed=1)
    assert abs(f - TARGET) < 1e-3


def test_local_search_depends_on_starting_point():
    r = local_runs(n_starts=30, seed=0)
    r_kms = r[:, 0] * 4.740470
    ok = np.abs(r_kms - TARGET) < 1e-2
    assert 0 < ok.sum() < len(r)          # some starts succeed, some do not
    assert r_kms.min() < TARGET + 1e-2    # the best local run finds the minimum
