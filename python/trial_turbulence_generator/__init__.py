"""Reusable CFD building blocks for the turbulence-generator project."""

from trial_turbulence_generator.core.domain import CartesianGrid, Grid1D
from trial_turbulence_generator.equations.burgers import ViscousBurgers1D
from trial_turbulence_generator.equations.euler import Euler1D

__all__ = [
    "CartesianGrid",
    "Euler1D",
    "Grid1D",
    "ViscousBurgers1D",
]
