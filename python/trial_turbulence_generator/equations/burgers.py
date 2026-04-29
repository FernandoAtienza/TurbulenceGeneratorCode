from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ViscousBurgers1D:
    """Scalar viscous Burgers equation, ``u_t + (u^2/2)_x = nu u_xx``."""

    viscosity: float

    @property
    def num_equations(self) -> int:
        return 1

    def flux(self, state: np.ndarray) -> np.ndarray:
        return 0.5 * state**2

    def max_wave_speed(self, state: np.ndarray) -> float:
        return float(np.max(np.abs(state)))

    def diffusion(self, second_derivative: np.ndarray) -> np.ndarray:
        return self.viscosity * second_derivative
