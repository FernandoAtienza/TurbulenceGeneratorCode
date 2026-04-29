from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trial_turbulence_generator.equations.euler import Euler1D


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


def interface_masks_with_joint(node_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shock_edge = node_mask & np.roll(node_mask, -1)
    smooth_edge = (~node_mask) & (~np.roll(node_mask, -1))
    joint_edge = ~(shock_edge | smooth_edge)
    return shock_edge, smooth_edge, joint_edge


@dataclass(frozen=True)
class EulerShockSensor:
    """Compression and relative-jump sensor used by the validated Euler scripts."""

    equation: Euler1D
    dx: float
    width: int = 4
    compression_threshold: float = 2.5
    jump_threshold: float = 0.04
    boundary_guard: int = 0

    def mask(self, conserved: np.ndarray) -> np.ndarray:
        rho, velocity, pressure = self.equation.primitive_from_conservative(conserved)
        theta = (np.roll(velocity, -1) - np.roll(velocity, 1)) / (2.0 * self.dx)
        theta_rms = np.sqrt(np.mean(theta**2))

        compression = np.zeros_like(theta, dtype=bool)
        if theta_rms > 1e-12:
            compression = theta < -self.compression_threshold * theta_rms

        density_jump = relative_jump_sensor(rho) > self.jump_threshold
        pressure_jump = relative_jump_sensor(pressure) > self.jump_threshold
        energy_jump = relative_jump_sensor(self.equation.internal_energy(rho, pressure)) > self.jump_threshold

        shock_nodes = compression | density_jump | pressure_jump | energy_jump
        if self.boundary_guard > 0:
            shock_nodes[: self.boundary_guard] = False
            shock_nodes[-self.boundary_guard :] = False
        return dilate_periodic_mask(shock_nodes, self.width)
