from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BurgersEquation:
    """1D viscous Burgers equation: u_t + (u^2 / 2)_x = nu u_xx."""

    viscosity: float

    def flux(self, u: np.ndarray) -> np.ndarray:
        return 0.5 * u**2

    def max_wave_speed(self, u: np.ndarray) -> float:
        return float(np.max(np.abs(u)))


@dataclass(frozen=True)
class EulerEquation:
    """1D ideal-gas Euler equation in conservative variables."""

    gamma: float = 1.4
    gas_constant: float = 1.0
    rho_floor: float = 1e-10
    pressure_floor: float = 1e-10

    def primitive_from_conservative(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.maximum(q[0], self.rho_floor)
        u = q[1] / rho
        pressure = (self.gamma - 1.0) * (q[2] - 0.5 * rho * u**2)
        return rho, u, np.maximum(pressure, self.pressure_floor)

    def conservative_from_primitive(
        self,
        rho: np.ndarray,
        u: np.ndarray,
        pressure: np.ndarray,
    ) -> np.ndarray:
        rho = np.maximum(rho, self.rho_floor)
        pressure = np.maximum(pressure, self.pressure_floor)
        q = np.zeros((3, rho.size), dtype=np.result_type(rho, u, pressure))
        q[0] = rho
        q[1] = rho * u
        q[2] = pressure / (self.gamma - 1.0) + 0.5 * rho * u**2
        return q

    def enforce_physical_state(self, q: np.ndarray) -> np.ndarray:
        q_fixed = np.array(q, copy=True)
        rho = np.maximum(q_fixed[0], self.rho_floor)
        u = q_fixed[1] / rho
        kinetic = 0.5 * rho * u**2
        pressure = (self.gamma - 1.0) * (q_fixed[2] - kinetic)

        q_fixed[0] = rho
        q_fixed[1] = rho * u
        q_fixed[2] = np.where(
            pressure > self.pressure_floor,
            q_fixed[2],
            self.pressure_floor / (self.gamma - 1.0) + kinetic,
        )
        return q_fixed

    def flux(self, q: np.ndarray) -> np.ndarray:
        rho, u, pressure = self.primitive_from_conservative(q)
        flux = np.zeros_like(q)
        flux[0] = q[1]
        flux[1] = rho * u**2 + pressure
        flux[2] = (q[2] + pressure) * u
        return flux

    def max_wave_speed(self, q: np.ndarray) -> float:
        rho, u, pressure = self.primitive_from_conservative(q)
        sound_speed = np.sqrt(self.gamma * pressure / rho)
        return float(np.max(np.abs(u) + sound_speed))

    def internal_energy(self, rho: np.ndarray, pressure: np.ndarray) -> np.ndarray:
        return pressure / (rho * (self.gamma - 1.0))

    def roe_eigenvectors(self, q_left: np.ndarray, q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rho_l, u_l, p_l = self.primitive_from_conservative(q_left[:, None])
        rho_r, u_r, p_r = self.primitive_from_conservative(q_right[:, None])
        rho_l, u_l, p_l = rho_l[0], u_l[0], p_l[0]
        rho_r, u_r, p_r = rho_r[0], u_r[0], p_r[0]

        h_l = (q_left[2] + p_l) / rho_l
        h_r = (q_right[2] + p_r) / rho_r
        sqrt_l = np.sqrt(rho_l)
        sqrt_r = np.sqrt(rho_r)
        u_roe = (sqrt_l * u_l + sqrt_r * u_r) / (sqrt_l + sqrt_r)
        h_roe = (sqrt_l * h_l + sqrt_r * h_r) / (sqrt_l + sqrt_r)

        kinetic = 0.5 * u_roe**2
        c2 = max((self.gamma - 1.0) * (h_roe - kinetic), self.pressure_floor)
        c = np.sqrt(c2)
        r_matrix = np.array(
            [
                [1.0, 1.0, 1.0],
                [u_roe - c, u_roe, u_roe + c],
                [h_roe - u_roe * c, kinetic, h_roe + u_roe * c],
            ]
        )

        gm1 = self.gamma - 1.0
        l_matrix = np.array(
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
        return l_matrix, r_matrix
