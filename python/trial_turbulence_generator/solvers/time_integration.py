from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from trial_turbulence_generator.core.boundary import BoundaryCondition, PeriodicBC

Array = np.ndarray
RHSFunction = Callable[[Array], Array]
Projection = Callable[[Array], Array]
PostStep = Callable[[int, Array], Array]


@dataclass(frozen=True)
class SSPRK3:
    """Third-order strong-stability-preserving Runge-Kutta scheme."""

    def step(
        self,
        state: Array,
        rhs: RHSFunction,
        dt: float,
        project: Projection | None = None,
    ) -> Array:
        project_state = project if project is not None else (lambda values: values)

        q0 = state
        q1 = project_state(q0 + dt * rhs(q0))
        q2 = project_state(0.75 * q0 + 0.25 * (q1 + dt * rhs(q1)))
        return project_state((1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + dt * rhs(q2)))


@dataclass
class SimulationResult:
    times: np.ndarray
    states: list[Array]

    @property
    def final_state(self) -> Array:
        return self.states[-1]


@dataclass
class ExplicitSolver1D:
    """Small orchestration class for explicit method-of-lines solvers."""

    state: Array
    rhs: RHSFunction
    dt: float
    boundary_condition: BoundaryCondition = field(default_factory=PeriodicBC)
    integrator: SSPRK3 = field(default_factory=SSPRK3)
    project: Projection | None = None
    post_step_hooks: Iterable[PostStep] = field(default_factory=list)
    boundary_axis: int = -1

    def __post_init__(self) -> None:
        self.state = self._project_and_bound(self.state)
        self.post_step_hooks = list(self.post_step_hooks)

    def _project_and_bound(self, state: Array) -> Array:
        out = self.project(state) if self.project is not None else state
        return self.boundary_condition.apply(out, axis=self.boundary_axis)

    def step(self, step_index: int) -> Array:
        self.state = self.integrator.step(
            self.state,
            self.rhs,
            self.dt,
            project=self._project_and_bound,
        )
        for hook in self.post_step_hooks:
            self.state = self._project_and_bound(hook(step_index, self.state))
        return self.state

    def run(self, t_end: float, output_times: Iterable[float] | None = None) -> SimulationResult:
        if t_end < 0.0:
            raise ValueError("t_end must be non-negative")

        if output_times is None:
            output_times = [t_end]
        output_times = np.array(list(output_times), dtype=float)
        output_times.sort()

        num_steps = int(np.ceil(t_end / self.dt)) if t_end > 0.0 else 0
        if num_steps > 0:
            self.dt = t_end / num_steps

        history: list[Array] = []
        times: list[float] = []
        next_output = 0

        while next_output < len(output_times) and output_times[next_output] <= 0.0:
            history.append(np.copy(self.state))
            times.append(0.0)
            next_output += 1

        for step_index in range(num_steps):
            self.step(step_index)
            current_time = (step_index + 1) * self.dt
            while (
                next_output < len(output_times)
                and current_time + 0.5 * self.dt >= output_times[next_output]
            ):
                history.append(np.copy(self.state))
                times.append(float(output_times[next_output]))
                next_output += 1

        while next_output < len(output_times):
            history.append(np.copy(self.state))
            times.append(t_end)
            next_output += 1

        return SimulationResult(np.array(times), history)
