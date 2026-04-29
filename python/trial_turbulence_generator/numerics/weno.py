from __future__ import annotations

import numpy as np


def weno7_flux(v1, v2, v3, v4, v5, v6, v7, eps: float = 1e-10):
    """Seventh-order WENO interpolation from four cubic substencils."""

    q0 = -(1 / 4) * v1 + (13 / 12) * v2 - (23 / 12) * v3 + (25 / 12) * v4
    q1 = (1 / 12) * v2 - (5 / 12) * v3 + (13 / 12) * v4 + (1 / 4) * v5
    q2 = -(1 / 12) * v3 + (7 / 12) * v4 + (7 / 12) * v5 - (1 / 12) * v6
    q3 = (1 / 4) * v4 + (13 / 12) * v5 - (5 / 12) * v6 + (1 / 12) * v7

    is0 = (
        v1 * (544 * v1 - 3882 * v2 + 4642 * v3 - 1854 * v4)
        + v2 * (7043 * v2 - 17246 * v3 + 7042 * v4)
        + v3 * (11003 * v3 - 9402 * v4)
        + 2107 * v4**2
    )
    is1 = (
        v2 * (267 * v2 - 1642 * v3 + 1602 * v4 - 494 * v5)
        + v3 * (2843 * v3 - 5966 * v4 + 1922 * v5)
        + v4 * (3443 * v4 - 2522 * v5)
        + 547 * v5**2
    )
    is2 = (
        v3 * (547 * v3 - 2522 * v4 + 1922 * v5 - 494 * v6)
        + v4 * (3443 * v4 - 5966 * v5 + 1602 * v6)
        + v5 * (2843 * v5 - 1642 * v6)
        + 267 * v6**2
    )
    is3 = (
        v4 * (2107 * v4 - 9402 * v5 + 7042 * v6 - 1854 * v7)
        + v5 * (11003 * v5 - 17246 * v6 + 4642 * v7)
        + v6 * (7043 * v6 - 3882 * v7)
        + 547 * v7**2
    )

    alpha0 = (1 / 35) / (eps + is0) ** 2
    alpha1 = (12 / 35) / (eps + is1) ** 2
    alpha2 = (18 / 35) / (eps + is2) ** 2
    alpha3 = (4 / 35) / (eps + is3) ** 2
    alpha_sum = alpha0 + alpha1 + alpha2 + alpha3
    return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2 + alpha3 * q3) / alpha_sum


def scalar_lax_friedrichs_weno7_flux(
    state: np.ndarray,
    physical_flux: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Periodic scalar interface flux ``F_{i+1/2}`` using LF splitting."""

    f_plus = 0.5 * (physical_flux + alpha * state)
    f_minus = 0.5 * (physical_flux - alpha * state)

    plus_half = weno7_flux(
        np.roll(f_plus, 3),
        np.roll(f_plus, 2),
        np.roll(f_plus, 1),
        f_plus,
        np.roll(f_plus, -1),
        np.roll(f_plus, -2),
        np.roll(f_plus, -3),
    )
    minus_half = weno7_flux(
        np.roll(f_minus, -4),
        np.roll(f_minus, -3),
        np.roll(f_minus, -2),
        np.roll(f_minus, -1),
        f_minus,
        np.roll(f_minus, 1),
        np.roll(f_minus, 2),
    )
    return plus_half + minus_half


def weno5_js_left(v0, v1, v2, v3, v4, eps: float = 1e-6):
    beta0 = (13.0 / 12.0) * (v0 - 2.0 * v1 + v2) ** 2
    beta0 += 0.25 * (v0 - 4.0 * v1 + 3.0 * v2) ** 2
    beta1 = (13.0 / 12.0) * (v1 - 2.0 * v2 + v3) ** 2
    beta1 += 0.25 * (v1 - v3) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v3 + v4) ** 2
    beta2 += 0.25 * (3.0 * v2 - 4.0 * v3 + v4) ** 2

    a0 = 0.1 / (eps + beta0) ** 2
    a1 = 0.6 / (eps + beta1) ** 2
    a2 = 0.3 / (eps + beta2) ** 2
    asum = a0 + a1 + a2

    p0 = (2.0 * v0 - 7.0 * v1 + 11.0 * v2) / 6.0
    p1 = (-v1 + 5.0 * v2 + 2.0 * v3) / 6.0
    p2 = (2.0 * v2 + 5.0 * v3 - v4) / 6.0
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def weno5_js_reconstruct_flux(f_plus: np.ndarray, f_minus: np.ndarray) -> np.ndarray:
    """Non-periodic WENO5-JS flux used for reference 1D shock-tube solutions."""

    ghost = 3
    n = f_plus.size
    fp_pad = np.pad(f_plus, (ghost, ghost), mode="edge")
    fm_pad = np.pad(f_minus, (ghost, ghost), mode="edge")

    flux_plus = weno5_js_left(
        fp_pad[0 : n + 1],
        fp_pad[1 : n + 2],
        fp_pad[2 : n + 3],
        fp_pad[3 : n + 4],
        fp_pad[4 : n + 5],
    )
    flux_minus = weno5_js_left(
        fm_pad[5 : n + 6],
        fm_pad[4 : n + 5],
        fm_pad[3 : n + 4],
        fm_pad[2 : n + 3],
        fm_pad[1 : n + 2],
    )
    return flux_plus + flux_minus
