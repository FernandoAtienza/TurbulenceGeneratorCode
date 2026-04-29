from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BoundaryCondition:
    """Base class for boundary operations on state arrays."""

    def apply(self, state: np.ndarray, axis: int = -1) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class PeriodicBC(BoundaryCondition):
    """Periodic boundary condition.

    Periodic compact/WENO operators use ``np.roll`` internally, so applying this
    boundary condition is intentionally a copy/no-op for non-ghosted arrays.
    """

    def apply(self, state: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.array(state, copy=True)


@dataclass(frozen=True)
class ExtrapolationBC(BoundaryCondition):
    """Zero-gradient boundary condition using guard cells inside the solve domain."""

    guard_cells: int

    def __post_init__(self) -> None:
        if self.guard_cells < 0:
            raise ValueError("guard_cells must be non-negative")

    def apply(self, state: np.ndarray, axis: int = -1) -> np.ndarray:
        arr = np.array(state, copy=True)
        if self.guard_cells == 0:
            return arr

        axis = axis % arr.ndim
        if arr.shape[axis] <= 2 * self.guard_cells:
            raise ValueError("axis is too short for the requested guard width")

        arr = np.moveaxis(arr, axis, -1)
        guard = self.guard_cells
        arr[..., :guard] = arr[..., guard][..., None]
        arr[..., -guard:] = arr[..., -guard - 1][..., None]
        return np.moveaxis(arr, -1, axis)
