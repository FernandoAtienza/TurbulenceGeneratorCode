from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from OOP.domain import Domain1D
from OOP.equations import BurgersEquation, EulerEquation
from OOP.sensors import EulerShockSensor, dilate_periodic_mask, interface_masks, relative_jump_sensor


def periodic_diags(values: list[float], offsets: list[int], size: int) -> sp.csc_matrix:
    matrix = sp.diags(values, offsets, shape=(size, size)).tolil()
    for value, offset in zip(values, offsets):
        if offset > 0:
            for row in range(offset):
                matrix[size - offset + row, row] = value
        elif offset < 0:
            for row in range(-offset):
                matrix[row, size + offset + row] = value
    return matrix.tocsc()


def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    eps = 1e-10
    q0 = -(1 / 4) * v1 + (13 / 12) * v2 - (23 / 12) * v3 + (25 / 12) * v4
    q1 = (1 / 12) * v2 - (5 / 12) * v3 + (13 / 12) * v4 + (1 / 4) * v5
    q2 = -(1 / 12) * v3 + (7 / 12) * v4 + (7 / 12) * v5 - (1 / 12) * v6
    q3 = (1 / 4) * v4 + (13 / 12) * v5 - (5 / 12) * v6 + (1 / 12) * v7

    is0 = (
        v1 * (544 * v1 - 3882 * v2 + 4642 * v3 - 1854 * v4)
        + v2 * (7043 * v2 - 17246 * v3 + 7042 * v4)
        + v3 * (11003 * v3 - 9402 * v4)
        + 2107 * v4**2
    )
    is1 = (
        v2 * (267 * v2 - 1642 * v3 + 1602 * v4 - 494 * v5)
        + v3 * (2843 * v3 - 5966 * v4 + 1922 * v5)
        + v4 * (3443 * v4 - 2522 * v5)
        + 547 * v5**2
    )
    is2 = (
        v3 * (547 * v3 - 2522 * v4 + 1922 * v5 - 494 * v6)
        + v4 * (3443 * v4 - 5966 * v5 + 1602 * v6)
        + v5 * (2843 * v5 - 1642 * v6)
        + 267 * v6**2
    )
    is3 = (
        v4 * (2107 * v4 - 9402 * v5 + 7042 * v6 - 1854 * v7)
        + v5 * (11003 * v5 - 17246 * v6 + 4642 * v7)
        + v6 * (7043 * v6 - 3882 * v7)
        + 547 * v7**2
    )

    a0 = (1 / 35) / (eps + is0) ** 2
    a1 = (12 / 35) / (eps + is1) ** 2
    a2 = (18 / 35) / (eps + is2) ** 2
    a3 = (4 / 35) / (eps + is3) ** 2
    asum = a0 + a1 + a2 + a3
    return (a0 * q0 + a1 * q1 + a2 * q2 + a3 * q3) / asum


def smooth_compact_flux(point_flux: np.ndarray) -> np.ndarray:
    a1_c = 25 / 32
    b1_c = 1 / 20
    c1_c = -1 / 480
    return (
        c1_c * (np.roll(point_flux, -3) + np.roll(point_flux, 2))
        + (b1_c + c1_c) * (np.roll(point_flux, -2) + np.roll(point_flux, 1))
        + (a1_c + b1_c + c1_c) * (np.roll(point_flux, -1) + point_flux)
    )


def second_derivative_6th(values: np.ndarray, dx: float) -> np.ndarray:
    return (
        (1 / 90) * (np.roll(values, -3) + np.roll(values, 3))
        - (3 / 20) * (np.roll(values, -2) + np.roll(values, 2))
        + (3 / 2) * (np.roll(values, -1) + np.roll(values, 1))
        - (49 / 18) * values
    ) / dx**2


@dataclass
class CompactDerivative:
    domain: Domain1D
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        matrix = periodic_diags([self.alpha, 1.0, self.alpha], [-1, 0, 1], self.domain.nx)
        self.solve_matrix = spla.factorized(matrix)

    def from_interface_flux(self, interface_flux: np.ndarray) -> np.ndarray:
        return self.solve_matrix((interface_flux - np.roll(interface_flux, 1)) / self.domain.dx)


@dataclass
class BurgersHybridSpatialOperator:
    domain: Domain1D
    equation: BurgersEquation
    sensor_width: int = 4
    jump_threshold: float = 0.04

    def __post_init__(self) -> None:
        self.compact = CompactDerivative(self.domain)

    def shock_mask(self, u: np.ndarray) -> np.ndarray:
        return dilate_periodic_mask(relative_jump_sensor(u) > self.jump_threshold, self.sensor_width)

    def weno_flux(self, u: np.ndarray) -> np.ndarray:
        f = self.equation.flux(u)
        alpha = self.equation.max_wave_speed(u)
        f_plus = 0.5 * (f + alpha * u)
        f_minus = 0.5 * (f - alpha * u)
        return weno7_flux(
            np.roll(f_plus, 3),
            np.roll(f_plus, 2),
            np.roll(f_plus, 1),
            f_plus,
            np.roll(f_plus, -1),
            np.roll(f_plus, -2),
            np.roll(f_plus, -3),
        ) + weno7_flux(
            np.roll(f_minus, -4),
            np.roll(f_minus, -3),
            np.roll(f_minus, -2),
            np.roll(f_minus, -1),
            f_minus,
            np.roll(f_minus, 1),
            np.roll(f_minus, 2),
        )

    def rhs(self, u: np.ndarray) -> np.ndarray:
        f = self.equation.flux(u)
        compact_flux = smooth_compact_flux(f)
        weno_raw = self.weno_flux(u)
        weno_flux_hat = self.compact.alpha * np.roll(weno_raw, -1) + weno_raw
        weno_flux_hat += self.compact.alpha * np.roll(weno_raw, 1)

        weno_edge, smooth_edge = interface_masks(self.shock_mask(u))
        hybrid_flux = np.empty_like(u)
        hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
        hybrid_flux[weno_edge] = weno_flux_hat[weno_edge]

        advection = self.compact.from_interface_flux(hybrid_flux)
        diffusion = self.equation.viscosity * second_derivative_6th(u, self.domain.dx)
        return -advection + diffusion


@dataclass
class EulerHybridSpatialOperator:
    domain: Domain1D
    equation: EulerEquation
    sensor: EulerShockSensor

    def __post_init__(self) -> None:
        self.compact = CompactDerivative(self.domain)

    def characteristic_weno_flux(
        self,
        q: np.ndarray,
        flux: np.ndarray,
        alpha: float,
        required_edges: np.ndarray,
    ) -> np.ndarray:
        f_half = np.zeros_like(q)
        f_plus = 0.5 * (flux + alpha * q)
        f_minus = 0.5 * (flux - alpha * q)
        indices = np.flatnonzero(dilate_periodic_mask(required_edges, 1))

        for i in indices:
            ip1 = (i + 1) % self.domain.nx
            left_matrix, right_matrix = self.equation.roe_eigenvectors(q[:, i], q[:, ip1])
            plus_stencil = np.array(
                [left_matrix @ f_plus[:, (i + offset) % self.domain.nx] for offset in range(-3, 4)]
            )
            minus_stencil = np.array(
                [left_matrix @ f_minus[:, (i + offset) % self.domain.nx] for offset in range(4, -3, -1)]
            )

            char_flux = np.empty(3)
            for component in range(3):
                char_flux[component] = weno7_flux(*plus_stencil[:, component])
                char_flux[component] += weno7_flux(*minus_stencil[:, component])
            f_half[:, i] = right_matrix @ char_flux

        return f_half

    def rhs(self, q: np.ndarray) -> np.ndarray:
        q_safe = self.equation.enforce_physical_state(q)
        flux = self.equation.flux(q_safe)
        alpha = self.equation.max_wave_speed(q_safe)
        weno_edge, smooth_edge = interface_masks(self.sensor.detect(q_safe))
        weno_raw = self.characteristic_weno_flux(q_safe, flux, alpha, weno_edge)

        rhs = np.zeros_like(q_safe)
        for component in range(3):
            compact_flux = smooth_compact_flux(flux[component])
            weno_flux_hat = self.compact.alpha * np.roll(weno_raw[component], -1)
            weno_flux_hat += weno_raw[component]
            weno_flux_hat += self.compact.alpha * np.roll(weno_raw[component], 1)

            hybrid_flux = np.empty_like(flux[component])
            hybrid_flux[smooth_edge] = compact_flux[smooth_edge]
            hybrid_flux[weno_edge] = weno_flux_hat[weno_edge]
            rhs[component] = -self.compact.from_interface_flux(hybrid_flux)

        return rhs
