from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trial_turbulence_generator.core.domain import Grid1D
from trial_turbulence_generator.equations.burgers import ViscousBurgers1D
from trial_turbulence_generator.equations.euler import Euler1D
from trial_turbulence_generator.numerics.compact import (
    CompactFluxDerivative1D,
    second_derivative_6th,
    smooth_compact_flux_8th,
)
from trial_turbulence_generator.numerics.sensors import (
    EulerShockSensor,
    dilate_periodic_mask,
    interface_masks,
    interface_masks_with_joint,
    relative_jump_sensor,
)
from trial_turbulence_generator.numerics.weno import scalar_lax_friedrichs_weno7_flux, weno7_flux


@dataclass
class HybridBurgersOperator1D:
    """Hybrid compact/WENO RHS for one-dimensional viscous Burgers."""

    grid: Grid1D
    equation: ViscousBurgers1D
    jump_threshold: float = 0.04
    sensor_width: int = 4

    def __post_init__(self) -> None:
        self.compact = CompactFluxDerivative1D(self.grid.num_cells, self.grid.dx)

    def discontinuity_mask(self, state: np.ndarray) -> np.ndarray:
        base = relative_jump_sensor(state) > self.jump_threshold
        return dilate_periodic_mask(base, self.sensor_width)

    def rhs(self, state: np.ndarray, shock_mask: np.ndarray | None = None) -> np.ndarray:
        if shock_mask is None:
            shock_mask = self.discontinuity_mask(state)

        point_flux = self.equation.flux(state)
        compact_flux = smooth_compact_flux_8th(point_flux)

        alpha = self.equation.max_wave_speed(state)
        weno_flux = scalar_lax_friedrichs_weno7_flux(state, point_flux, alpha)
        weno_flux_hat = (
            self.compact.alpha * np.roll(weno_flux, -1)
            + weno_flux
            + self.compact.alpha * np.roll(weno_flux, 1)
        )

        shock_edge, smooth_edge, joint_edge = interface_masks_with_joint(shock_mask)
        hybrid_flux = np.empty_like(state)
        hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
        hybrid_flux[shock_edge] = weno_flux_hat[shock_edge]
        hybrid_flux[joint_edge] = 0.5 * (compact_flux[joint_edge] + weno_flux_hat[joint_edge])

        advection = self.compact.derivative_from_interface_flux(hybrid_flux)
        diffusion = self.equation.diffusion(second_derivative_6th(state, self.grid.dx))
        return -advection + diffusion


@dataclass
class HybridEulerOperator1D:
    """Hybrid compact/WENO RHS for one-dimensional Euler systems."""

    grid: Grid1D
    equation: Euler1D
    sensor: EulerShockSensor

    def __post_init__(self) -> None:
        self.compact = CompactFluxDerivative1D(self.grid.num_cells, self.grid.dx)

    def discontinuity_mask(self, conserved: np.ndarray) -> np.ndarray:
        return self.sensor.mask(conserved)

    def characteristic_weno7_flux(
        self,
        conserved: np.ndarray,
        physical_flux: np.ndarray,
        alpha: float,
        required_edges: np.ndarray | None = None,
    ) -> np.ndarray:
        num_cells = self.grid.num_cells
        flux_half = np.zeros_like(conserved)
        f_plus = 0.5 * (physical_flux + alpha * conserved)
        f_minus = 0.5 * (physical_flux - alpha * conserved)

        if required_edges is None:
            indices = range(num_cells)
        else:
            indices = np.flatnonzero(dilate_periodic_mask(required_edges, 1))

        for i in indices:
            ip1 = (i + 1) % num_cells
            left_vectors, right_vectors = self.equation.roe_eigenvectors(
                conserved[:, i],
                conserved[:, ip1],
            )
            plus_stencil = np.array(
                [left_vectors @ f_plus[:, (i + offset) % num_cells] for offset in range(-3, 4)]
            )
            minus_stencil = np.array(
                [left_vectors @ f_minus[:, (i + offset) % num_cells] for offset in range(4, -3, -1)]
            )

            flux_characteristic = np.empty(self.equation.num_equations)
            for component in range(self.equation.num_equations):
                flux_characteristic[component] = (
                    weno7_flux(*plus_stencil[:, component])
                    + weno7_flux(*minus_stencil[:, component])
                )
            flux_half[:, i] = right_vectors @ flux_characteristic

        return flux_half

    def rhs(self, conserved: np.ndarray, shock_mask: np.ndarray | None = None) -> np.ndarray:
        if shock_mask is None:
            shock_mask = self.discontinuity_mask(conserved)

        safe_state = self.equation.enforce_physical_state(conserved)
        physical_flux = self.equation.flux(safe_state)
        alpha = self.equation.max_wave_speed(safe_state)

        weno_edge, smooth_edge = interface_masks(shock_mask)
        weno_raw = self.characteristic_weno7_flux(
            safe_state,
            physical_flux,
            alpha,
            required_edges=weno_edge,
        )

        advection = np.zeros_like(safe_state)
        for component in range(self.equation.num_equations):
            compact_flux = smooth_compact_flux_8th(physical_flux[component])
            weno_flux_hat = (
                self.compact.alpha * np.roll(weno_raw[component], -1)
                + weno_raw[component]
                + self.compact.alpha * np.roll(weno_raw[component], 1)
            )

            hybrid_flux = np.empty_like(physical_flux[component])
            hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
            hybrid_flux[weno_edge] = weno_flux_hat[weno_edge]
            advection[component] = self.compact.derivative_from_interface_flux(hybrid_flux)

        return -advection
