from __future__ import annotations

import numpy as np

from trial_turbulence_generator.equations.euler import Euler1D


def burgers_sine_initial_condition(x: np.ndarray) -> np.ndarray:
    return -np.sin(np.pi * x)


def sod_initial_condition(
    x: np.ndarray,
    equation: Euler1D,
    discontinuity_location: float = 0.5,
    left_state: tuple[float, float, float] = (1.0, 0.0, 1.0),
    right_state: tuple[float, float, float] = (0.125, 0.0, 0.1),
) -> np.ndarray:
    rho_l, u_l, p_l = left_state
    rho_r, u_r, p_r = right_state

    rho = np.where(x < discontinuity_location, rho_l, rho_r)
    velocity = np.where(x < discontinuity_location, u_l, u_r)
    pressure = np.where(x < discontinuity_location, p_l, p_r)
    return equation.conservative_from_primitive(rho, velocity, pressure)


def shu_osher_initial_condition(
    x: np.ndarray,
    equation: Euler1D,
    shock_position: float = 1.0,
    left_velocity: float = 2.629,
    left_pressure: float = 10.333,
    left_temperature: float = 2.679,
    right_velocity: float = 0.0,
    right_pressure: float = 1.0,
    entropy_amplitude: float = 0.2,
    entropy_wavenumber: float = 5.0,
    entropy_phase: float = 25.0,
) -> np.ndarray:
    left_density = left_pressure / (equation.gas_constant * left_temperature)
    right_temperature = 1.0 / (
        1.0 + entropy_amplitude * np.sin(entropy_wavenumber * x - entropy_phase)
    )
    right_density = right_pressure / (equation.gas_constant * right_temperature)

    rho = np.where(x < shock_position, left_density, right_density)
    velocity = np.where(x < shock_position, left_velocity, right_velocity)
    pressure = np.where(x < shock_position, left_pressure, right_pressure)
    return equation.conservative_from_primitive(rho, velocity, pressure)
