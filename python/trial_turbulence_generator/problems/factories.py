from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trial_turbulence_generator.core.boundary import BoundaryCondition, ExtrapolationBC, PeriodicBC
from trial_turbulence_generator.core.domain import Grid1D
from trial_turbulence_generator.equations.burgers import ViscousBurgers1D
from trial_turbulence_generator.equations.euler import Euler1D
from trial_turbulence_generator.numerics.hyperviscosity import WangHyperviscosity1D
from trial_turbulence_generator.numerics.sensors import EulerShockSensor
from trial_turbulence_generator.numerics.spatial import HybridBurgersOperator1D, HybridEulerOperator1D
from trial_turbulence_generator.problems.initial_conditions import (
    burgers_sine_initial_condition,
    shu_osher_initial_condition,
    sod_initial_condition,
)
from trial_turbulence_generator.solvers.time_integration import ExplicitSolver1D


@dataclass
class ProblemSetup:
    name: str
    grid: Grid1D
    equation: object
    operator: object
    solver: ExplicitSolver1D
    hyperviscosity: WangHyperviscosity1D | None = None


def _adjust_dt(dt: float, t_end: float | None) -> float:
    if t_end is None:
        return dt
    num_steps = int(np.ceil(t_end / dt))
    return t_end / num_steps


def _hyperviscosity_hook(operator, hyperviscosity: WangHyperviscosity1D):
    def hook(step_index: int, state: np.ndarray) -> np.ndarray:
        if (step_index + 1) % hyperviscosity.interval != 0:
            return state
        return hyperviscosity.apply(state, operator.discontinuity_mask(state))

    return hook


def make_burgers_hybrid_setup(
    nx: int = 30,
    x_min: float = -1.0,
    x_max: float = 1.0,
    dt: float = 0.001,
    t_end: float | None = None,
    viscosity: float = 0.01 / np.pi,
    use_hyperviscosity: bool = True,
    hyperviscosity_coefficient: float = 0.01,
    hyperviscosity_interval: int = 5,
) -> ProblemSetup:
    grid = Grid1D(x_min, x_max, nx)
    equation = ViscousBurgers1D(viscosity=viscosity)
    operator = HybridBurgersOperator1D(grid=grid, equation=equation)
    state = burgers_sine_initial_condition(grid.cell_centers)
    dt = _adjust_dt(dt, t_end)

    hyperviscosity = None
    hooks = []
    if use_hyperviscosity:
        hyperviscosity = WangHyperviscosity1D(
            grid.num_cells,
            grid.dx,
            dt,
            coefficient=hyperviscosity_coefficient,
            interval=hyperviscosity_interval,
            blend_joints=True,
        )
        hooks.append(_hyperviscosity_hook(operator, hyperviscosity))

    solver = ExplicitSolver1D(
        state=state,
        rhs=operator.rhs,
        dt=dt,
        boundary_condition=PeriodicBC(),
        post_step_hooks=hooks,
    )
    return ProblemSetup("1D viscous Burgers", grid, equation, operator, solver, hyperviscosity)


def make_sod_hybrid_setup(
    nx: int = 300,
    x_min: float = -1.0,
    x_max: float = 2.0,
    dt: float = 0.0005,
    t_end: float | None = 0.2,
    gamma: float = 1.4,
    hyperviscosity_coefficient: float = 0.02,
    hyperviscosity_interval: int = 5,
    sensor_width: int = 4,
    compression_threshold: float = 2.5,
    jump_threshold: float = 0.035,
    boundary_condition: BoundaryCondition | None = None,
) -> ProblemSetup:
    grid = Grid1D(x_min, x_max, nx)
    equation = Euler1D(gamma=gamma)
    state = sod_initial_condition(grid.cell_centers, equation)
    dt = _adjust_dt(dt, t_end)

    sensor = EulerShockSensor(
        equation,
        grid.dx,
        width=sensor_width,
        compression_threshold=compression_threshold,
        jump_threshold=jump_threshold,
    )
    operator = HybridEulerOperator1D(grid=grid, equation=equation, sensor=sensor)
    hyperviscosity = WangHyperviscosity1D(
        grid.num_cells,
        grid.dx,
        dt,
        coefficient=hyperviscosity_coefficient,
        interval=hyperviscosity_interval,
    )
    solver = ExplicitSolver1D(
        state=state,
        rhs=operator.rhs,
        dt=dt,
        boundary_condition=boundary_condition or PeriodicBC(),
        project=equation.enforce_physical_state,
        post_step_hooks=[_hyperviscosity_hook(operator, hyperviscosity)],
    )
    return ProblemSetup("Sod shock tube", grid, equation, operator, solver, hyperviscosity)


def make_shu_osher_hybrid_setup(
    nx_physical: int = 200,
    physical_min: float = 0.0,
    physical_max: float = 10.0,
    solve_min: float = -10.0,
    solve_max: float = 20.0,
    cfl: float = 0.22,
    t_end: float = 1.8,
    gamma: float = 1.4,
    hyperviscosity_coefficient: float = 0.02,
    hyperviscosity_interval: int = 5,
    sensor_width: int = 4,
    compression_threshold: float = 2.5,
    jump_threshold: float = 0.04,
    boundary_guard: int = 8,
) -> ProblemSetup:
    dx = (physical_max - physical_min) / nx_physical
    grid = Grid1D.from_spacing(solve_min, solve_max, dx)
    equation = Euler1D(gamma=gamma)
    state = shu_osher_initial_condition(grid.cell_centers, equation)

    max_speed = equation.max_wave_speed(state)
    dt = cfl * grid.dx / max_speed
    dt = _adjust_dt(dt, t_end)

    sensor = EulerShockSensor(
        equation,
        grid.dx,
        width=sensor_width,
        compression_threshold=compression_threshold,
        jump_threshold=jump_threshold,
        boundary_guard=boundary_guard,
    )
    operator = HybridEulerOperator1D(grid=grid, equation=equation, sensor=sensor)
    hyperviscosity = WangHyperviscosity1D(
        grid.num_cells,
        grid.dx,
        dt,
        coefficient=hyperviscosity_coefficient,
        interval=hyperviscosity_interval,
    )
    solver = ExplicitSolver1D(
        state=state,
        rhs=operator.rhs,
        dt=dt,
        boundary_condition=ExtrapolationBC(boundary_guard),
        project=equation.enforce_physical_state,
        post_step_hooks=[_hyperviscosity_hook(operator, hyperviscosity)],
    )
    return ProblemSetup("Shu-Osher shock tube", grid, equation, operator, solver, hyperviscosity)
