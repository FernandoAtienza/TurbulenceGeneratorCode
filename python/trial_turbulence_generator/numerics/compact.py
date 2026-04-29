from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


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


def smooth_compact_flux_8th(point_flux: np.ndarray) -> np.ndarray:
    """Interface flux used by the validated eighth-order compact scheme."""

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


def second_derivative_8th(values: np.ndarray, dx: float) -> np.ndarray:
    return (
        -(1 / 560) * np.roll(values, -4)
        + (8 / 315) * np.roll(values, -3)
        - (1 / 5) * np.roll(values, -2)
        + (8 / 5) * np.roll(values, -1)
        - (205 / 72) * values
        + (8 / 5) * np.roll(values, 1)
        - (1 / 5) * np.roll(values, 2)
        + (8 / 315) * np.roll(values, 3)
        - (1 / 560) * np.roll(values, 4)
    ) / dx**2


@dataclass
class CompactFluxDerivative1D:
    """Periodic compact derivative from interface flux differences."""

    num_cells: int
    dx: float
    alpha: float = 3.0 / 8.0

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if self.dx <= 0.0:
            raise ValueError("dx must be positive")
        matrix = periodic_diags(
            [self.alpha, 1.0, self.alpha],
            [-1, 0, 1],
            self.num_cells,
        )
        self._solve = spla.factorized(matrix)

    def derivative_from_interface_flux(self, interface_flux: np.ndarray) -> np.ndarray:
        return self._solve((interface_flux - np.roll(interface_flux, 1)) / self.dx)

    def smooth_derivative(self, point_flux: np.ndarray) -> np.ndarray:
        return self.derivative_from_interface_flux(smooth_compact_flux_8th(point_flux))
