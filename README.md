# Numerical integrators for the Kepler problem

A comparison of four numerical integrators applied to the Kepler problem, rewritten in Python from the MATLAB programs developed for my M.Sc. thesis in Mathematics on symplectic integrators.

Implemented methods:

- Explicit Euler (order 1)
- Symplectic Euler (order 1)
- Implicit midpoint (order 2), solved with `scipy.optimize.fsolve`
- Störmer–Verlet (order 2)

## Problem

Kepler problem in normalised units (mu = 1, semi-major axis a = 1) with eccentricity e = 0.7, starting from pericentre. The state is y = (x, y, vx, vy). The conserved quantities are the energy E = -0.5 and the angular momentum L = sqrt(1 - e^2). The reference solution is computed with `scipy.integrate.solve_ivp` (DOP853, tolerances 1e-13).

## Structure

- `kepler.py`: force, initial state, energy, angular momentum, the four integrators, `integrate` and the reference solution.
- `convergence.py`: convergence study. Computes the global error at t = 3 for several step sizes, estimates the observed order with a log-log fit and saves `figures/convergence.png`.
- `orbits.py`: orbit comparison for e = 0.7; saves `figures/orbits.png`.
- `tests/test_kepler.py`: automated tests with pytest.
- `requirements.txt`: required libraries.

## Usage

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 convergence.py
python3 orbits.py
pytest -v
```

## Results

### Convergence

Observed convergence orders (global error at t = 3):

| Method | Observed order |
|---|---|
| Explicit Euler | 1.00 |
| Symplectic Euler | 0.97 |
| Implicit midpoint | 2.00 |
| Störmer–Verlet | 2.00 |

![Convergence](figures/convergence.png)

Note: explicit Euler uses much smaller steps (20,000 to 160,000 steps) than the other methods, because with few steps it is not in the asymptotic regime and the observed order is underestimated.

### Orbits

Orbits obtained with e = 0.7, 5 orbits and 200 steps per orbit, compared with the exact ellipse (dashed). All start from pericentre.

![Orbits](figures/orbits.png)

With this relatively coarse step:

- **Explicit Euler** diverges: the energy grows at every step and the orbit opens up until it leaves the ellipse.
- **Symplectic Euler** stays bounded, but the numerical orbit is distorted with respect to the exact ellipse.
- **Implicit midpoint** keeps the shape and size of the ellipse, but the major axis rotates at every revolution (a phase error, absent in the exact problem).
- **Störmer–Verlet** is the closest to the exact orbit, with a very small precession.

Symplectic and symmetric methods do not show the systematic energy drift of explicit Euler: the dominant error is a phase error, and the energy stays bounded.

## Tests

The five automated tests check:

- energy and angular momentum of the initial state against the theoretical values;
- order 2 of Störmer–Verlet and order 1 of symplectic Euler, within a tolerance;
- conservation of angular momentum with Störmer–Verlet over 10 orbits;
- bounded energy (error below 1e-3) with Störmer–Verlet over 100 orbits.

## Possible extensions

- Systematic comparison with the results of the original MATLAB programs.
- Study of other eccentricities and longer integration times.
- Planar circular restricted three-body problem.