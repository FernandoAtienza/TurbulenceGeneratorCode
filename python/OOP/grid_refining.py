from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from OOP.domain import Domain1D


@dataclass(frozen=True)
class GridRefiner1D:
    """Small helper for comparing the same case on finer/coarser 1D grids."""

    domain: Domain1D

    def refined_domain(self, factor: int) -> Domain1D:
        if factor <= 0:
            raise ValueError("factor must be positive")
        return Domain1D(
            self.domain.x_min,
            self.domain.x_max,
            self.domain.nx * factor,
            endpoint=self.domain.endpoint,
        )

    def interpolate_to(self, values: np.ndarray, target: Domain1D) -> np.ndarray:
        """Linearly interpolate values from this domain to another 1D domain."""

        source_x = self.domain.x
        target_x = target.x
        if values.ndim == 1:
            return np.interp(target_x, source_x, values)

        interpolated = np.empty((values.shape[0], target.nx), dtype=values.dtype)
        for component in range(values.shape[0]):
            interpolated[component] = np.interp(target_x, source_x, values[component])
        return interpolated

    def coarsen_by_average(self, values: np.ndarray, factor: int) -> np.ndarray:
        if factor <= 0:
            raise ValueError("factor must be positive")
        usable = (values.shape[-1] // factor) * factor
        trimmed = values[..., :usable]
        return trimmed.reshape(*trimmed.shape[:-1], usable // factor, factor).mean(axis=-1)
