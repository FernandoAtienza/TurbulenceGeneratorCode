from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trial_turbulence_generator.problems.factories import make_burgers_hybrid_setup


def cole_hopf_solution(x: np.ndarray, t: float, viscosity: float) -> np.ndarray:
    if t == 0.0:
        return -np.sin(np.pi * x)

    roots, weights = hermgauss(30)
    solution = np.zeros_like(x)
    for i, xi in enumerate(x):
        g = np.sqrt(4 * viscosity * t) * roots
        y = xi - g
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * viscosity))
        solution[i] = np.sum(weights * (-np.sin(np.pi * y)) * f_y) / np.sum(weights * f_y)
    return solution


def main() -> None:
    t_end = 1.0
    setup = make_burgers_hybrid_setup(t_end=t_end)
    result = setup.solver.run(t_end=t_end, output_times=[0.0, t_end])

    x = setup.grid.cell_centers
    numerical = result.final_state
    exact = cole_hopf_solution(x, t_end, setup.equation.viscosity)

    plt.figure(figsize=(8, 5))
    plt.plot(x, exact, "k-", lw=1.5, label="Cole-Hopf")
    plt.scatter(x, numerical, color="tab:red", s=25, label="Hybrid OOP")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.title(f"1D viscous Burgers at t = {t_end:.3f}")
    plt.show()


if __name__ == "__main__":
    main()
