"""Reusable numerical methods."""

from trial_turbulence_generator.numerics.compact import CompactFluxDerivative1D
from trial_turbulence_generator.numerics.hyperviscosity import WangHyperviscosity1D
from trial_turbulence_generator.numerics.sensors import EulerShockSensor
from trial_turbulence_generator.numerics.spatial import HybridBurgersOperator1D, HybridEulerOperator1D

__all__ = [
    "CompactFluxDerivative1D",
    "EulerShockSensor",
    "HybridBurgersOperator1D",
    "HybridEulerOperator1D",
    "WangHyperviscosity1D",
]
