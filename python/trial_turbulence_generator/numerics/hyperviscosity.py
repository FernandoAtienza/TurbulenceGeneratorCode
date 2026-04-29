from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from trial_turbulence_generator.numerics.compact import periodic_diags
from trial_turbulence_generator.numerics.sensors import interface_masks_with_joint


@dataclass
class WangHyperviscosity1D:
    """Semi-implicit Wang et al. hyperviscosity operator used by the hybrid scheme."""

    num_cells: int
    dx: float
    dt: float
    coefficient: float = 0.02
    interval: int = 5
    blend_joints: bool = False

    def __post_init__(self) -> None:
        if self.interval <= 0:
            raise ValueError("interval must be positive")
        self.dt_hyp = self.interval * self.dt

        self.a2_h = 4 / 9
        self.b2_h = 1 / 36
        self.a2_t = 20 / 27
        self.b2_t = 25 / 216

        self.a3_h = 344 / 1179
        self.b3_h = (38 * self.a3_h - 9) / 214
        self.a3_t = (696 - 1191 * self.a3_h) / 428
        self.b3_t = (1227 * self.a3_h - 147) / 1070

        self.A_hyp = periodic_diags(
            [self.b2_h, self.a2_h, 1.0, self.a2_h, self.b2_h],
            [-2, -1, 0, 1, 2],
            self.num_cells,
        )
        self.solve_A = spla.factorized(self.A_hyp)

        self.B_hyp = periodic_diags(
            [
                -self.b2_t / self.dx,
                -self.a2_t / self.dx,
                self.a2_t / self.dx,
                self.b2_t / self.dx,
            ],
            [-2, -1, 1, 2],
            self.num_cells,
        )

        self.C_hyp = periodic_diags(
            [self.b3_h, self.a3_h, 1.0, self.a3_h, self.b3_h],
            [-2, -1, 0, 1, 2],
            self.num_cells,
        )
        self.solve_C = spla.factorized(self.C_hyp)

        self.D_hyp = periodic_diags(
            [
                self.b3_t / self.dx**2,
                self.a3_t / self.dx**2,
                -2 * (self.a3_t + self.b3_t) / self.dx**2,
                self.a3_t / self.dx**2,
                self.b3_t / self.dx**2,
            ],
            [-2, -1, 0, 1, 2],
            self.num_cells,
        )
        self.solve_implicit = spla.factorized(
            self.C_hyp - self.dt_hyp * self.coefficient * self.D_hyp
        )

    def apply_scalar(self, values: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        values_x = self.solve_A(self.B_hyp @ values)
        g_half = (
            self.b2_t * np.roll(values_x, 1)
            + (self.a2_t + self.b2_t) * values_x
            + (self.a2_t + self.b2_t) * np.roll(values_x, -1)
            + self.b2_t * np.roll(values_x, -2)
        ) / self.dx

        d_half = (
            -self.b3_t * np.roll(values, 1)
            - (self.a3_t + self.b3_t) * values
            + (self.a3_t + self.b3_t) * np.roll(values, -1)
            + self.b3_t * np.roll(values, -2)
        ) / self.dx**2
        h_half = self.A_hyp @ self.solve_C(d_half)

        shock_edge, smooth_edge, joint_edge = interface_masks_with_joint(shock_mask)
        blended = np.empty_like(g_half)
        blended[smooth_edge] = g_half[smooth_edge]
        blended[shock_edge] = h_half[shock_edge]
        if self.blend_joints:
            blended[joint_edge] = 0.5 * (g_half[joint_edge] + h_half[joint_edge])
        else:
            blended[joint_edge] = h_half[joint_edge]

        explicit_term = self.solve_A(blended - np.roll(blended, 1))
        rhs = values - self.dt_hyp * self.coefficient * explicit_term
        return self.solve_implicit(self.C_hyp @ rhs)

    def apply(self, state: np.ndarray, shock_mask: np.ndarray) -> np.ndarray:
        if state.ndim == 1:
            return self.apply_scalar(state, shock_mask)

        filtered = np.empty_like(state)
        for component in range(state.shape[0]):
            filtered[component] = self.apply_scalar(state[component], shock_mask)
        return filtered
