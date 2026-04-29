from __future__ import annotations

import numpy as np

from OOP.equations import EulerEquation


def burgers_sine(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def sod_shock_tube(x: np.ndarray, equation: EulerEquation) -> np.ndarray:
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    pressure = np.where(x < 0.5, 1.0, 0.1)
    return equation.conservative_from_primitive(rho, u, pressure)


def shu_osher_shock_tube(x: np.ndarray, equation: EulerEquation) -> np.ndarray:
    left_pressure = 10.333
    left_temperature = 2.679
    left_density = left_pressure / (equation.gas_constant * left_temperature)

    right_pressure = 1.0
    right_temperature = 1.0 / (1.0 + 0.2 * np.sin(5.0 * x - 25.0))
    right_density = right_pressure / (equation.gas_constant * right_temperature)

    rho = np.where(x < 1.0, left_density, right_density)
    u = np.where(x < 1.0, 2.629, 0.0)
    pressure = np.where(x < 1.0, left_pressure, right_pressure)
    return equation.conservative_from_primitive(rho, u, pressure)
