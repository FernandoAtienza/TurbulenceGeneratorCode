from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from OOP.domain import Domain1D
from OOP.sensors import interface_masks
from OOP.spatial_operator import periodic_diags


@dataclass
class WangHyperviscosity:
    """Semi-implicit hyperviscosity used in the current hybrid 1D scripts."""

    domain: Domain1D
    dt: float
    mn: float = 0.02
    interval: int = 5

    def __post_init__(self) -> None:
        dx = self.domain.dx
        nx = self.domain.nx
        self.dt_hyp = self.interval * self.dt

        self.a2_t = 20 / 27
        self.b2_t = 25 / 216
        self.a3_h = 344 / 1179
        self.b3_h = (38 * self.a3_h - 9) / 214
        self.a3_t = (696 - 1191 * self.a3_h) / 428
        self.b3_t = (1227 * self.a3_h - 147) / 1070

        self.a_matrix = periodic_diags([1 / 36, 4 / 9, 1.0, 4 / 9, 1 / 36], [-2, -1, 0, 1, 2], nx)
        self.solve_a = spla.factorized(self.a_matrix)

        self.b_matrix = periodic_diags(
            [-self.b2_t / dx, -self.a2_t / dx, self.a2_t / dx, self.b2_t / dx],
            [-2, -1, 1, 2],
            nx,
        )

        self.c_matrix = periodic_diags(
            [self.b3_h, self.a3_h, 1.0, self.a3_h, self.b3_h],
            [-2, -1, 0, 1, 2],
            nx,
        )
        self.solve_c = spla.factorized(self.c_matrix)

        self.d_matrix = periodic_diags(
            [
                self.b3_t / dx**2,
                self.a3_t / dx**2,
                -2 * (self.a3_t + self.b3_t) / dx**2,
                self.a3_t / dx**2,
                self.b3_t / dx**2,
            ],
            [-2, -1, 0, 1, 2],
            nx,
        )
        self.solve_implicit = spla.factorized(self.c_matrix - self.dt_hyp * self.mn * self.d_matrix)

    def apply_scalar(self, values: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        dx = self.domain.dx
        values_x = self.solve_a(self.b_matrix @ values)
        g_half = (
            self.b2_t * np.roll(values_x, 1)
            + (self.a2_t + self.b2_t) * values_x
            + (self.a2_t + self.b2_t) * np.roll(values_x, -1)
            + self.b2_t * np.roll(values_x, -2)
        ) / dx

        d_half = (
            -self.b3_t * np.roll(values, 1)
            - (self.a3_t + self.b3_t) * values
            + (self.a3_t + self.b3_t) * np.roll(values, -1)
            + self.b3_t * np.roll(values, -2)
        ) / dx**2
        h_half = self.a_matrix @ self.solve_c(d_half)

        shock_edge, smooth_edge = interface_masks(shock_mask)
        blended = np.empty_like(values)
        blended[smooth_edge] = g_half[smooth_edge]
        blended[shock_edge] = h_half[shock_edge]

        explicit = self.solve_a(blended - np.roll(blended, 1))
        rhs = values - self.dt_hyp * self.mn * explicit
        return self.solve_implicit(self.c_matrix @ rhs)

    def apply(self, state: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            return self.apply_scalar(state, shock_mask)

        updated = np.empty_like(state)
        for component in range(state.shape[0]):
            updated[component] = self.apply_scalar(state[component], shock_mask)
        return updated
