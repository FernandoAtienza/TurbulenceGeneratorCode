from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trial_turbulence_generator.problems.factories import make_shu_osher_hybrid_setup


def main() -> None:
    t_end = 1.8
    setup = make_shu_osher_hybrid_setup(t_end=t_end)
    result = setup.solver.run(t_end=t_end, output_times=[t_end])

    x = setup.grid.cell_centers
    mask = setup.grid.mask_between(0.0, 10.0)
    rho, velocity, pressure = setup.equation.primitive_from_conservative(result.final_state[:, mask])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(x[mask], pressure, color="tab:blue", lw=1.2)
    axes[0].set_ylabel("P")
    axes[1].plot(x[mask], rho, color="tab:green", lw=1.2)
    axes[1].set_ylabel("rho")
    axes[2].plot(x[mask], velocity, color="tab:red", lw=1.2)
    axes[2].set_ylabel("u")
    axes[2].set_xlabel("x")

    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.45)

    fig.suptitle(f"Shu-Osher shock tube, OOP solver, t = {t_end:.3f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
