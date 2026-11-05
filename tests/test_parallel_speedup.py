import numpy as np

from parallel_speedup import heavy_cost, run
from trajectory_optimisation import cost


def test_heavy_cost_equals_cost():
    x = (1.8, 0.7)
    assert heavy_cost(x, reps=3) == cost(x)


def test_parallel_run_gives_same_result_as_serial():
    """With deferred updating and a fixed seed, the number of workers must not change the result."""
    _, f1 = run(workers=1, reps=1, generations=5, popsize=10)
    _, f2 = run(workers=2, reps=1, generations=5, popsize=10)
    assert np.isclose(f1, f2, rtol=0, atol=1e-12)
