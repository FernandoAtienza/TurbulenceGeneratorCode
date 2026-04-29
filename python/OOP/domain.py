from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain1D:
    """Uniform 1D domain used by the current Burgers and shock-tube scripts."""

    x_min: float
    x_max: float
    nx: int
    endpoint: bool = False

    def __post_init__(self) -> None:
        if self.nx <= 0:
            raise ValueError("nx must be positive")
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")

    @property
    def length(self) -> float:
        return self.x_max - self.x_min

    @property
    def dx(self) -> float:
        intervals = self.nx - 1 if self.endpoint else self.nx
        return self.length / intervals

    @property
    def x(self) -> np.ndarray:
        if self.endpoint:
            return np.linspace(self.x_min, self.x_max, self.nx)
        return self.x_min + self.dx * np.arange(self.nx)

    def mask(self, x_min: float, x_max: float) -> np.ndarray:
        return (self.x >= x_min) & (self.x <= x_max)

    @classmethod
    def from_dx(cls, x_min: float, x_max: float, dx: float) -> "Domain1D":
        if dx <= 0.0:
            raise ValueError("dx must be positive")
        nx = int(round((x_max - x_min) / dx))
        return cls(x_min=x_min, x_max=x_max, nx=nx)
