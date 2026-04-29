from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Grid1D:
    """Uniform one-dimensional grid.

    The grid stores cell centers by default with the right endpoint excluded,
    which matches the periodic discretizations already used in the validation
    scripts.
    """

    x_min: float
    x_max: float
    num_cells: int
    endpoint: bool = False

    def __post_init__(self) -> None:
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        intervals = self.num_cells - 1 if self.endpoint else self.num_cells
        return self.length / intervals

    @property
    def shape(self) -> tuple[int]:
        return (self.num_cells,)

    @property
    def cell_centers(self) -> np.ndarray:
        if self.endpoint:
            return np.linspace(self.x_min, self.x_max, self.num_cells)
        return self.x_min + self.dx * np.arange(self.num_cells)

    @property
    def nodes(self) -> np.ndarray:
        return self.cell_centers

    def mask_between(self, x_min: float, x_max: float) -> np.ndarray:
        x = self.cell_centers
        return (x >= x_min) & (x <= x_max)

    @classmethod
    def from_spacing(cls, x_min: float, x_max: float, dx: float) -> "Grid1D":
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        num_cells = int(round((x_max - x_min) / dx))
        return cls(x_min=x_min, x_max=x_max, num_cells=num_cells)


@dataclass(frozen=True)
class CartesianGrid:
    """Uniform Cartesian grid for future 2D/3D Euler and Navier-Stokes cases."""

    bounds: Sequence[tuple[float, float]]
    shape: Sequence[int]
    endpoint: bool = False

    def __post_init__(self) -> None:
        if len(self.bounds) != len(self.shape):
            raise ValueError("bounds and shape must have the same dimensionality")
        if len(self.bounds) not in (1, 2, 3):
            raise ValueError("CartesianGrid currently supports 1D, 2D, and 3D")
        for axis_bounds, n in zip(self.bounds, self.shape):
            if n <= 0:
                raise ValueError("all shape entries must be positive")
            if axis_bounds[1] <= axis_bounds[0]:
                raise ValueError("each upper bound must exceed its lower bound")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        spacings = []
        for (a, b), n in zip(self.bounds, self.shape):
            intervals = n - 1 if self.endpoint else n
            spacings.append((b - a) / intervals)
        return tuple(spacings)

    @property
    def axes(self) -> tuple[np.ndarray, ...]:
        axes = []
        for (a, _b), n, dx in zip(self.bounds, self.shape, self.spacing):
            if self.endpoint:
                axes.append(np.linspace(a, _b, n))
            else:
                axes.append(a + dx * np.arange(n))
        return tuple(axes)

    @property
    def mesh(self) -> tuple[np.ndarray, ...]:
        return np.meshgrid(*self.axes, indexing="ij")

    def as_1d(self) -> Grid1D:
        if self.ndim != 1:
            raise ValueError("Only a one-dimensional CartesianGrid can be converted to Grid1D")
        return Grid1D(*self.bounds[0], int(self.shape[0]), endpoint=self.endpoint)
