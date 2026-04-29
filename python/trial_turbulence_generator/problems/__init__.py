"""Initial conditions and validated benchmark problem factories."""

from trial_turbulence_generator.problems.initial_conditions import (
    burgers_sine_initial_condition,
    shu_osher_initial_condition,
    sod_initial_condition,
)
from trial_turbulence_generator.problems.factories import (
    ProblemSetup,
    make_burgers_hybrid_setup,
    make_shu_osher_hybrid_setup,
    make_sod_hybrid_setup,
)

__all__ = [
    "ProblemSetup",
    "burgers_sine_initial_condition",
    "make_burgers_hybrid_setup",
    "make_shu_osher_hybrid_setup",
    "make_sod_hybrid_setup",
    "shu_osher_initial_condition",
    "sod_initial_condition",
]
