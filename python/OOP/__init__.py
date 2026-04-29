"""Small 1D-only OOP structure for the validated Burgers and shock-tube cases."""

from OOP.domain import Domain1D
from OOP.equations import BurgersEquation, EulerEquation
from OOP.time_operator import SSPRK3, TimeOperator

__all__ = [
    "BurgersEquation",
    "Domain1D",
    "EulerEquation",
    "SSPRK3",
    "TimeOperator",
]
