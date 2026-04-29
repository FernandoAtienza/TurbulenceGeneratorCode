from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Euler1D:
    """Ideal-gas one-dimensional Euler equations in conservative form."""

    gamma: float = 1.4
    gas_constant: float = 1.0
    rho_floor: float = 1e-10
    pressure_floor: float = 1e-10

    @property
    def num_equations(self) -> int:
        return 3

    def primitive_from_conservative(self, conserved: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.maximum(conserved[0], self.rho_floor)
        velocity = conserved[1] / rho
        total_energy = conserved[2]
        pressure = (self.gamma - 1.0) * (total_energy - 0.5 * rho * velocity**2)
        pressure = np.maximum(pressure, self.pressure_floor)
        return rho, velocity, pressure

    def conservative_from_primitive(
        self,
        rho: np.ndarray,
        velocity: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        rho = np.maximum(rho, self.rho_floor)
        pressure = np.maximum(pressure, self.pressure_floor)

        conserved = np.zeros((self.num_equations, rho.size), dtype=np.result_type(rho, velocity, pressure))
        conserved[0] = rho
        conserved[1] = rho * velocity
        conserved[2] = pressure / (self.gamma - 1.0) + 0.5 * rho * velocity**2
        return conserved

    def enforce_physical_state(self, conserved: np.ndarray) -> np.ndarray:
        fixed = np.array(conserved, copy=True)
        rho = np.maximum(fixed[0], self.rho_floor)
        velocity = fixed[1] / rho
        kinetic = 0.5 * rho * velocity**2
        pressure = (self.gamma - 1.0) * (fixed[2] - kinetic)

        fixed[0] = rho
        fixed[1] = rho * velocity
        fixed[2] = np.where(
            pressure > self.pressure_floor,
            fixed[2],
            self.pressure_floor / (self.gamma - 1.0) + kinetic,
        )
        return fixed

    def flux(self, conserved: np.ndarray) -> np.ndarray:
        rho, velocity, pressure = self.primitive_from_conservative(conserved)
        total_energy = conserved[2]

        flux = np.zeros_like(conserved)
        flux[0] = conserved[1]
        flux[1] = rho * velocity**2 + pressure
        flux[2] = (total_energy + pressure) * velocity
        return flux

    def sound_speed(self, conserved: np.ndarray) -> np.ndarray:
        rho, _velocity, pressure = self.primitive_from_conservative(conserved)
        return np.sqrt(self.gamma * pressure / rho)

    def max_wave_speed(self, conserved: np.ndarray) -> float:
        _rho, velocity, _pressure = self.primitive_from_conservative(conserved)
        return float(np.max(np.abs(velocity) + self.sound_speed(conserved)))

    def internal_energy(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return pressure / (rho * (self.gamma - 1.0))

    def entropy(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return np.log(pressure / rho**self.gamma)

    def roe_eigenvectors(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho_l, u_l, p_l = self.primitive_from_conservative(left[:, None])
        rho_r, u_r, p_r = self.primitive_from_conservative(right[:, None])
        rho_l, u_l, p_l = rho_l[0], u_l[0], p_l[0]
        rho_r, u_r, p_r = rho_r[0], u_r[0], p_r[0]

        h_l = (left[2] + p_l) / rho_l
        h_r = (right[2] + p_r) / rho_r
        sqrt_l = np.sqrt(rho_l)
        sqrt_r = np.sqrt(rho_r)
        denom = sqrt_l + sqrt_r

        u_roe = (sqrt_l * u_l + sqrt_r * u_r) / denom
        h_roe = (sqrt_l * h_l + sqrt_r * h_r) / denom
        kinetic = 0.5 * u_roe**2
        c2 = max((self.gamma - 1.0) * (h_roe - kinetic), self.pressure_floor)
        c = np.sqrt(c2)

        right_vectors = np.array(
            [
                [1.0, 1.0, 1.0],
                [u_roe - c, u_roe, u_roe + c],
                [h_roe - u_roe * c, kinetic, h_roe + u_roe * c],
            ]
        )

        gm1 = self.gamma - 1.0
        left_vectors = np.array(
            [
                [
                    (gm1 * kinetic + u_roe * c) / (2.0 * c2),
                    -(gm1 * u_roe + c) / (2.0 * c2),
                    gm1 / (2.0 * c2),
                ],
                [1.0 - gm1 * kinetic / c2, gm1 * u_roe / c2, -gm1 / c2],
                [
                    (gm1 * kinetic - u_roe * c) / (2.0 * c2),
                    -(gm1 * u_roe - c) / (2.0 * c2),
                    gm1 / (2.0 * c2),
                ],
            ]
        )
        return left_vectors, right_vectors
