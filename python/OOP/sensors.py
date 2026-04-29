from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from OOP.equations import EulerEquation


def relative_jump_sensor(values: np.ndarray) -> np.ndarray:
    eps = 1e-14 + 1e-12 * np.max(np.abs(values))
    jump_r = np.abs(np.roll(values, -1) - values)
    jump_r /= np.abs(np.roll(values, -1)) + np.abs(values) + eps

    jump_l = np.abs(values - np.roll(values, 1))
    jump_l /= np.abs(values) + np.abs(np.roll(values, 1)) + eps

    curvature = np.abs(np.roll(values, -1) - 2.0 * values + np.roll(values, 1))
    curvature /= (
        np.abs(np.roll(values, -1))
        + 2.0 * np.abs(values)
        + np.abs(np.roll(values, 1))
        + eps
    )
    return np.maximum.reduce([jump_l, jump_r, curvature])


def dilate_periodic_mask(mask: np.ndarray, width: int) -> np.ndarray:
    expanded = np.array(mask, copy=True)
    for offset in range(1, width + 1):
        expanded |= np.roll(mask, offset)
        expanded |= np.roll(mask, -offset)
    return expanded


def interface_masks(node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weno_edge = node_mask | np.roll(node_mask, -1)
    smooth_edge = ~weno_edge
    return weno_edge, smooth_edge


@dataclass(frozen=True)
class EulerShockSensor:
    equation: EulerEquation
    dx: float
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    boundary_guard: int = 0

    def detect(self, q: np.ndarray) -> np.ndarray:
        rho, u, pressure = self.equation.primitive_from_conservative(q)
        theta = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * self.dx)
        theta_rms = np.sqrt(np.mean(theta**2))

        compression = np.zeros_like(theta, dtype=bool)
        if theta_rms > 1e-12:
            compression = theta < -self.compression_threshold * theta_rms

        density_jump = relative_jump_sensor(rho) > self.jump_threshold
        pressure_jump = relative_jump_sensor(pressure) > self.jump_threshold
        energy_jump = relative_jump_sensor(self.equation.internal_energy(rho, pressure))
        energy_jump = energy_jump > self.jump_threshold

        mask = compression | density_jump | pressure_jump | energy_jump
        if self.boundary_guard:
            mask[: self.boundary_guard] = False
            mask[-self.boundary_guard :] = False
        return dilate_periodic_mask(mask, self.width)
