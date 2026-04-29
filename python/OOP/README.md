# Simple 1D OOP Structure

This folder is a smaller OOP pass for only the cases already validated:

- 1D viscous Burgers equation.
- 1D Sod shock tube.
- 1D Shu-Osher shock tube.

It does not try to prepare for 2D/3D problems yet. The goal is to make the
current progression easy to follow.

## Files

- `domain.py`: `Domain1D`, a uniform 1D grid.
- `boundary_conditions.py`: periodic and extrapolated boundary conditions.
- `grid_refining.py`: simple refine/interpolate/coarsen helpers for 1D grids.
- `equations.py`: Burgers and Euler equation objects.
- `sensors.py`: shock and relative-jump sensors.
- `spatial_operator.py`: compact/WENO hybrid RHS operators.
- `hyperviscosity.py`: Wang semi-implicit hyperviscosity.
- `time_operator.py`: SSP-RK3 and the time loop.
- `initial_conditions.py`: the current Burgers, Sod, and Shu-Osher initial data.

## Minimal Burgers Setup

```python
from OOP_1D.domain import Domain1D
from OOP_1D.equations import BurgersEquation
from OOP_1D.initial_conditions import burgers_sine
from OOP_1D.spatial_operator import BurgersHybridSpatialOperator
from OOP_1D.time_operator import TimeOperator

domain = Domain1D(-1.0, 1.0, 30)
equation = BurgersEquation(viscosity=0.01 / 3.141592653589793)
operator = BurgersHybridSpatialOperator(domain, equation)

state0 = burgers_sine(domain.x)
solver = TimeOperator(state0, operator.rhs, dt=0.001)
times, states = solver.run(t_end=0.1)
```

## Minimal Shock-Tube Setup

```python
from OOP_1D.boundary_conditions import ExtrapolatedBoundary
from OOP_1D.domain import Domain1D
from OOP_1D.equations import EulerEquation
from OOP_1D.initial_conditions import sod_shock_tube
from OOP_1D.sensors import EulerShockSensor
from OOP_1D.spatial_operator import EulerHybridSpatialOperator
from OOP_1D.time_operator import TimeOperator

domain = Domain1D(-1.0, 2.0, 300)
equation = EulerEquation()
sensor = EulerShockSensor(equation, domain.dx, jump_threshold=0.035)
operator = EulerHybridSpatialOperator(domain, equation, sensor)

state0 = sod_shock_tube(domain.x, equation)
solver = TimeOperator(
    state0,
    operator.rhs,
    dt=0.0005,
    boundary=ExtrapolatedBoundary(guard_cells=0),
    state_filter=equation.enforce_physical_state,
)
times, states = solver.run(t_end=0.2)
```
