# Turbulence Generator OOP Package

This package stores reusable CFD pieces for the validated one-dimensional
cases and for future 2D/3D Euler, Navier-Stokes, turbulence, and Riemann
problem work.

## Layout

- `core/`: grid and boundary-condition objects.
- `equations/`: physical systems such as Burgers and ideal-gas Euler.
- `numerics/`: sensors, WENO reconstruction, compact operators, hybrid spatial
  operators, and Wang hyperviscosity.
- `solvers/`: time integration and solver orchestration.
- `problems/`: initial conditions and benchmark setup factories.
- `examples/`: runnable OOP examples for the current validated cases.

## Typical Use

```python
from turbulence_generator.problems.factories import make_shu_osher_hybrid_setup

t_end = 1.8
setup = make_shu_osher_hybrid_setup(t_end=t_end)
result = setup.solver.run(t_end=t_end, output_times=[t_end])

rho, u, p = setup.equation.primitive_from_conservative(result.final_state)
```

When running from the repository root, add `python` to `PYTHONPATH` or run the
example scripts from inside the `python` directory.

## Extension Points

- Add `Grid2D`/`Grid3D` specializations or use `CartesianGrid` directly for
  multi-dimensional work.
- Add new equation objects with `flux`, `max_wave_speed`, and optional
  `enforce_physical_state` methods.
- Add new boundary condition classes by implementing `BoundaryCondition.apply`.
- Add AMR/grid-refinement objects beside `core/domain.py`; they can own nested
  `CartesianGrid` patches and provide prolongation/restriction methods.
- Add 2D/3D spatial operators under `numerics/` while keeping the same solver
  orchestration in `solvers/`.
