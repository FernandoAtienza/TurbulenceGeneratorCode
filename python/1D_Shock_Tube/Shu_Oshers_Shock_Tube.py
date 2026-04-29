# ============================================================
# 1D Euler Equations: Shu-Osher Shock Tube
# Hybrid Scheme: 8th-Order Compact FD + WENO-7 + Hyperviscosity
# ============================================================

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
gamma = 1.4
rho_floor = 1e-10
P_floor = 1e-10

# Di Renzo et al. 2020, section 6.4.3: x in [0, 10], shock initially
# at x = 1. The right thermodynamic state is prescribed through
# T = [1 + 0.2 sin(5x - 25)]^-1, so rho = p / T for R = 1.
physical_min, physical_max = 0.0, 10.0
solve_min, solve_max = -10.0, 20.0
shock_initial_position = 1.0

R_gas = 1.0
left_u = 2.629
left_P = 10.333
left_T = 2.679
right_u = 0.0
right_P = 1.0
entropy_amplitude = 0.2
entropy_wavenumber = 5.0
entropy_phase = 25.0

nx_physical = int(os.getenv("SHU_OSHER_NX", "200"))
reference_nx_physical = int(os.getenv("SHU_OSHER_REFERENCE_NX", "2000"))
t_end = 1.8
cfl = 0.22

mn = 0.02
hyperviscosity_interval = 5
sensor_width = 4
compression_threshold = 2.5
jump_threshold = 0.04
boundary_guard = 8


def shu_osher_initial_condition(x_arr):
    left_rho = left_P / (R_gas * left_T)
    right_T = 1.0 / (
        1.0 + entropy_amplitude * np.sin(entropy_wavenumber * x_arr - entropy_phase)
    )
    right_rho = right_P / (R_gas * right_T)
    rho = np.where(x_arr < shock_initial_position, left_rho, right_rho)
    u = np.where(x_arr < shock_initial_position, left_u, right_u)
    P = np.where(x_arr < shock_initial_position, left_P, right_P)
    return rho, u, P


def primitive_from_conservative(Q_arr):
    rho_arr = np.maximum(Q_arr[0], rho_floor)
    u_arr = Q_arr[1] / rho_arr
    E_arr = Q_arr[2]
    P_arr = (gamma - 1.0) * (E_arr - 0.5 * rho_arr * u_arr**2)
    P_arr = np.maximum(P_arr, P_floor)
    return rho_arr, u_arr, P_arr


def conservative_from_primitive(rho_arr, u_arr, P_arr):
    rho_arr = np.maximum(rho_arr, rho_floor)
    P_arr = np.maximum(P_arr, P_floor)

    Q_arr = np.zeros((3, rho_arr.size))
    Q_arr[0] = rho_arr
    Q_arr[1] = rho_arr * u_arr
    Q_arr[2] = P_arr / (gamma - 1.0) + 0.5 * rho_arr * u_arr**2
    return Q_arr


def enforce_physical_state(Q_arr):
    Q_fixed = np.array(Q_arr, copy=True)
    rho_arr = np.maximum(Q_fixed[0], rho_floor)
    u_arr = Q_fixed[1] / rho_arr
    kinetic = 0.5 * rho_arr * u_arr**2
    P_arr = (gamma - 1.0) * (Q_fixed[2] - kinetic)

    Q_fixed[0] = rho_arr
    Q_fixed[1] = rho_arr * u_arr
    Q_fixed[2] = np.where(
        P_arr > P_floor,
        Q_fixed[2],
        P_floor / (gamma - 1.0) + kinetic,
    )
    return Q_fixed


def euler_flux(Q_arr):
    rho_arr, u_arr, P_arr = primitive_from_conservative(Q_arr)
    E_arr = Q_arr[2]

    F_arr = np.zeros_like(Q_arr)
    F_arr[0] = Q_arr[1]
    F_arr[1] = rho_arr * u_arr**2 + P_arr
    F_arr[2] = (E_arr + P_arr) * u_arr
    return F_arr


def internal_energy(rho_arr, P_arr):
    return P_arr / (rho_arr * (gamma - 1.0))


def entropy_function(rho_arr, P_arr):
    return np.log(P_arr / rho_arr**gamma)


def relative_jump_sensor(q_arr):
    eps = 1e-14 + 1e-12 * np.max(np.abs(q_arr))
    jump_r = np.abs(np.roll(q_arr, -1) - q_arr)
    jump_r /= np.abs(np.roll(q_arr, -1)) + np.abs(q_arr) + eps

    jump_l = np.abs(q_arr - np.roll(q_arr, 1))
    jump_l /= np.abs(q_arr) + np.abs(np.roll(q_arr, 1)) + eps

    curvature = np.abs(np.roll(q_arr, -1) - 2.0 * q_arr + np.roll(q_arr, 1))
    curvature /= (
        np.abs(np.roll(q_arr, -1))
        + 2.0 * np.abs(q_arr)
        + np.abs(np.roll(q_arr, 1))
        + eps
    )
    return np.maximum.reduce([jump_l, jump_r, curvature])


def dilate_periodic_mask(mask, width):
    expanded = mask.copy()
    for offset in range(1, width + 1):
        expanded |= np.roll(mask, offset)
        expanded |= np.roll(mask, -offset)
    return expanded


def interface_masks(node_mask):
    weno_edge = node_mask | np.roll(node_mask, -1)
    smooth_edge = ~weno_edge
    return weno_edge, smooth_edge


def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    eps = 1e-10
    q0 = -(1 / 4) * v1 + (13 / 12) * v2 - (23 / 12) * v3 + (25 / 12) * v4
    q1 = (1 / 12) * v2 - (5 / 12) * v3 + (13 / 12) * v4 + (1 / 4) * v5
    q2 = -(1 / 12) * v3 + (7 / 12) * v4 + (7 / 12) * v5 - (1 / 12) * v6
    q3 = (1 / 4) * v4 + (13 / 12) * v5 - (5 / 12) * v6 + (1 / 12) * v7

    IS0 = (
        v1 * (544 * v1 - 3882 * v2 + 4642 * v3 - 1854 * v4)
        + v2 * (7043 * v2 - 17246 * v3 + 7042 * v4)
        + v3 * (11003 * v3 - 9402 * v4)
        + 2107 * v4**2
    )
    IS1 = (
        v2 * (267 * v2 - 1642 * v3 + 1602 * v4 - 494 * v5)
        + v3 * (2843 * v3 - 5966 * v4 + 1922 * v5)
        + v4 * (3443 * v4 - 2522 * v5)
        + 547 * v5**2
    )
    IS2 = (
        v3 * (547 * v3 - 2522 * v4 + 1922 * v5 - 494 * v6)
        + v4 * (3443 * v4 - 5966 * v5 + 1602 * v6)
        + v5 * (2843 * v5 - 1642 * v6)
        + 267 * v6**2
    )
    IS3 = (
        v4 * (2107 * v4 - 9402 * v5 + 7042 * v6 - 1854 * v7)
        + v5 * (11003 * v5 - 17246 * v6 + 4642 * v7)
        + v6 * (7043 * v6 - 3882 * v7)
        + 547 * v7**2
    )

    alpha0 = (1 / 35) / (eps + IS0) ** 2
    alpha1 = (12 / 35) / (eps + IS1) ** 2
    alpha2 = (18 / 35) / (eps + IS2) ** 2
    alpha3 = (4 / 35) / (eps + IS3) ** 2
    sum_alpha = alpha0 + alpha1 + alpha2 + alpha3

    return (alpha0 * q0 + alpha1 * q1 + alpha2 * q2 + alpha3 * q3) / sum_alpha


def roe_eigenvectors(Q_left, Q_right):
    rho_l, u_l, P_l = primitive_from_conservative(Q_left[:, None])
    rho_r, u_r, P_r = primitive_from_conservative(Q_right[:, None])
    rho_l, u_l, P_l = rho_l[0], u_l[0], P_l[0]
    rho_r, u_r, P_r = rho_r[0], u_r[0], P_r[0]

    H_l = (Q_left[2] + P_l) / rho_l
    H_r = (Q_right[2] + P_r) / rho_r
    sqrt_l = np.sqrt(rho_l)
    sqrt_r = np.sqrt(rho_r)
    denom = sqrt_l + sqrt_r

    u_roe = (sqrt_l * u_l + sqrt_r * u_r) / denom
    H_roe = (sqrt_l * H_l + sqrt_r * H_r) / denom
    kinetic = 0.5 * u_roe**2
    c2 = max((gamma - 1.0) * (H_roe - kinetic), P_floor)
    c = np.sqrt(c2)

    R = np.array(
        [
            [1.0, 1.0, 1.0],
            [u_roe - c, u_roe, u_roe + c],
            [H_roe - u_roe * c, kinetic, H_roe + u_roe * c],
        ]
    )

    gm1 = gamma - 1.0
    L = np.array(
        [
            [
                (gm1 * kinetic + u_roe * c) / (2.0 * c2),
                -(gm1 * u_roe + c) / (2.0 * c2),
                gm1 / (2.0 * c2),
            ],
            [1.0 - gm1 * kinetic / c2, gm1 * u_roe / c2, -gm1 / c2],
            [
                (gm1 * kinetic - u_roe * c) / (2.0 * c2),
                -(gm1 * u_roe - c) / (2.0 * c2),
                gm1 / (2.0 * c2),
            ],
        ]
    )
    return L, R


class HybridEulerSolver:
    def __init__(self, nx_physical_cells, label):
        self.label = label
        self.dx = (physical_max - physical_min) / nx_physical_cells
        self.nx = int(round((solve_max - solve_min) / self.dx))
        self.x = solve_min + self.dx * np.arange(self.nx)

        rho, u, P = shu_osher_initial_condition(self.x)
        self.Q = conservative_from_primitive(rho, u, P)
        self.Q = self.apply_boundary(self.Q)

        rho0, u0, P0 = primitive_from_conservative(self.Q)
        max_wave_speed = np.max(np.abs(u0) + np.sqrt(gamma * P0 / rho0))
        self.dt = cfl * self.dx / max_wave_speed
        self.num_steps = int(np.ceil(t_end / self.dt))
        self.dt = t_end / self.num_steps
        self.dt_hyp = hyperviscosity_interval * self.dt

        self._build_matrices()

    def _build_matrices(self):
        nx = self.nx
        dx = self.dx

        self.alpha1_c = 3.0 / 8.0
        A_adv_lil = sp.diags(
            [self.alpha1_c, 1.0, self.alpha1_c],
            [-1, 0, 1],
            shape=(nx, nx),
        ).tolil()
        A_adv_lil[0, -1] = self.alpha1_c
        A_adv_lil[-1, 0] = self.alpha1_c
        self.solve_A_adv = spla.factorized(A_adv_lil.tocsc())

        self.a2_t = 20 / 27
        self.b2_t = 25 / 216
        self.a3_h = 344 / 1179
        self.b3_h = (38 * self.a3_h - 9) / 214
        self.a3_t = (696 - 1191 * self.a3_h) / 428
        self.b3_t = (1227 * self.a3_h - 147) / 1070

        A_hyp_lil = sp.diags(
            [1 / 36, 4 / 9, 1.0, 4 / 9, 1 / 36],
            [-2, -1, 0, 1, 2],
            shape=(nx, nx),
        ).tolil()
        A_hyp_lil[0, -1] = 4 / 9
        A_hyp_lil[0, -2] = 1 / 36
        A_hyp_lil[1, -1] = 1 / 36
        A_hyp_lil[-1, 0] = 4 / 9
        A_hyp_lil[-2, 0] = 1 / 36
        A_hyp_lil[-1, 1] = 1 / 36
        self.A_hyp_csc = A_hyp_lil.tocsc()
        self.solve_A_hyp = spla.factorized(self.A_hyp_csc)

        B_hyp_lil = sp.diags(
            [
                -self.b2_t / dx,
                -self.a2_t / dx,
                self.a2_t / dx,
                self.b2_t / dx,
            ],
            [-2, -1, 1, 2],
            shape=(nx, nx),
        ).tolil()
        B_hyp_lil[0, -1] = -self.a2_t / dx
        B_hyp_lil[0, -2] = -self.b2_t / dx
        B_hyp_lil[1, -1] = -self.b2_t / dx
        B_hyp_lil[-1, 0] = self.a2_t / dx
        B_hyp_lil[-2, 0] = self.b2_t / dx
        B_hyp_lil[-1, 1] = self.b2_t / dx
        self.B_hyp = B_hyp_lil.tocsc()

        C_hyp_lil = sp.diags(
            [self.b3_h, self.a3_h, 1.0, self.a3_h, self.b3_h],
            [-2, -1, 0, 1, 2],
            shape=(nx, nx),
        ).tolil()
        C_hyp_lil[0, -1] = self.a3_h
        C_hyp_lil[0, -2] = self.b3_h
        C_hyp_lil[1, -1] = self.b3_h
        C_hyp_lil[-1, 0] = self.a3_h
        C_hyp_lil[-2, 0] = self.b3_h
        C_hyp_lil[-1, 1] = self.b3_h
        self.C_hyp_csc = C_hyp_lil.tocsc()
        self.solve_C_hyp = spla.factorized(self.C_hyp_csc)

        D_hyp_lil = sp.diags(
            [
                self.b3_t / dx**2,
                self.a3_t / dx**2,
                -2 * (self.a3_t + self.b3_t) / dx**2,
                self.a3_t / dx**2,
                self.b3_t / dx**2,
            ],
            [-2, -1, 0, 1, 2],
            shape=(nx, nx),
        ).tolil()
        D_hyp_lil[0, -1] = self.a3_t / dx**2
        D_hyp_lil[0, -2] = self.b3_t / dx**2
        D_hyp_lil[1, -1] = self.b3_t / dx**2
        D_hyp_lil[-1, 0] = self.a3_t / dx**2
        D_hyp_lil[-2, 0] = self.b3_t / dx**2
        D_hyp_lil[-1, 1] = self.b3_t / dx**2
        self.D_hyp = D_hyp_lil.tocsc()
        self.solve_L_hyp = spla.factorized(
            self.C_hyp_csc - self.dt_hyp * mn * self.D_hyp
        )

    def apply_boundary(self, Q_arr):
        Q_bc = np.array(Q_arr, copy=True)
        Q_bc[:, :boundary_guard] = Q_bc[:, boundary_guard][:, None]
        Q_bc[:, -boundary_guard:] = Q_bc[:, -boundary_guard - 1][:, None]
        return Q_bc

    def discontinuity_mask(self, Q_arr):
        rho_arr, u_arr, P_arr = primitive_from_conservative(Q_arr)
        theta = (np.roll(u_arr, -1) - np.roll(u_arr, 1)) / (2.0 * self.dx)
        theta_rms = np.sqrt(np.mean(theta**2))

        compression = np.zeros_like(theta, dtype=bool)
        if theta_rms > 1e-12:
            compression = theta < -compression_threshold * theta_rms

        density_jump = relative_jump_sensor(rho_arr) > jump_threshold
        pressure_jump = relative_jump_sensor(P_arr) > jump_threshold
        energy_jump = relative_jump_sensor(internal_energy(rho_arr, P_arr)) > jump_threshold

        shock_nodes = compression | density_jump | pressure_jump | energy_jump
        shock_nodes[:boundary_guard] = False
        shock_nodes[-boundary_guard:] = False
        return dilate_periodic_mask(shock_nodes, sensor_width)

    def get_weno7_euler_flux_LLF(self, Q_arr, F_arr, alpha, required_edges):
        F_half = np.zeros_like(Q_arr)
        F_plus = 0.5 * (F_arr + alpha * Q_arr)
        F_minus = 0.5 * (F_arr - alpha * Q_arr)
        indices = np.flatnonzero(dilate_periodic_mask(required_edges, 1))

        for i in indices:
            ip1 = (i + 1) % self.nx
            L, R = roe_eigenvectors(Q_arr[:, i], Q_arr[:, ip1])

            plus_stencil = np.array(
                [L @ F_plus[:, (i + offset) % self.nx] for offset in range(-3, 4)]
            )
            minus_stencil = np.array(
                [L @ F_minus[:, (i + offset) % self.nx] for offset in range(4, -3, -1)]
            )

            flux_char = np.empty(3)
            for m in range(3):
                flux_char[m] = (
                    weno7_flux(*plus_stencil[:, m])
                    + weno7_flux(*minus_stencil[:, m])
                )

            F_half[:, i] = R @ flux_char

        return F_half

    def rhs_euler_hybrid(self, Q_curr, shock_mask):
        Q_safe = enforce_physical_state(Q_curr)
        rho_arr, u_arr, P_arr = primitive_from_conservative(Q_safe)
        F_curr = euler_flux(Q_safe)

        c_arr = np.sqrt(gamma * P_arr / rho_arr)
        alpha = float(np.max(np.abs(u_arr) + c_arr))

        a1_c = 25 / 32
        b1_c = 1 / 20
        c1_c = -1 / 480
        advection = np.zeros_like(Q_safe)
        weno_edge, smooth_edge = interface_masks(shock_mask)

        F_weno_raw = self.get_weno7_euler_flux_LLF(Q_safe, F_curr, alpha, weno_edge)

        for k in range(3):
            f = F_curr[k]
            F_comp = (
                c1_c * (np.roll(f, -3) + np.roll(f, 2))
                + (b1_c + c1_c) * (np.roll(f, -2) + np.roll(f, 1))
                + (a1_c + b1_c + c1_c) * (np.roll(f, -1) + f)
            )
            F_weno_hat = (
                self.alpha1_c * np.roll(F_weno_raw[k], -1)
                + F_weno_raw[k]
                + self.alpha1_c * np.roll(F_weno_raw[k], 1)
            )

            F_hybrid = np.empty_like(f)
            F_hybrid[smooth_edge] = F_comp[smooth_edge]
            F_hybrid[weno_edge] = F_weno_hat[weno_edge]

            advection[k] = self.solve_A_adv((F_hybrid - np.roll(F_hybrid, 1)) / self.dx)

        return -advection

    def apply_semi_implicit_hyperviscosity(self, u_current, shock_mask):
        u_x = self.solve_A_hyp(self.B_hyp @ u_current)
        G_half = (
            self.b2_t * np.roll(u_x, 1)
            + (self.a2_t + self.b2_t) * u_x
            + (self.a2_t + self.b2_t) * np.roll(u_x, -1)
            + self.b2_t * np.roll(u_x, -2)
        ) / self.dx
        D_half_u = (
            -self.b3_t * np.roll(u_current, 1)
            - (self.a3_t + self.b3_t) * u_current
            + (self.a3_t + self.b3_t) * np.roll(u_current, -1)
            + self.b3_t * np.roll(u_current, -2)
        ) / (self.dx**2)
        H_half = self.A_hyp_csc @ self.solve_C_hyp(D_half_u)

        shock_edge, smooth_edge = interface_masks(shock_mask)

        G_mod = np.empty_like(G_half)
        G_mod[smooth_edge] = G_half[smooth_edge]
        G_mod[shock_edge] = H_half[shock_edge]

        E = self.solve_A_hyp(G_mod - np.roll(G_mod, 1))
        rhs_implicit = u_current - self.dt_hyp * mn * E
        return self.solve_L_hyp(self.C_hyp_csc @ rhs_implicit)

    def apply_hyperviscosity_to_conserved(self, Q_curr, shock_mask):
        Q_visc = np.empty_like(Q_curr)
        for k in range(3):
            Q_visc[k] = self.apply_semi_implicit_hyperviscosity(Q_curr[k], shock_mask)
        return self.apply_boundary(enforce_physical_state(Q_visc))

    def step(self, step_index):
        shock_region = self.discontinuity_mask(self.Q)
        Q1 = enforce_physical_state(
            self.Q + self.dt * self.rhs_euler_hybrid(self.Q, shock_region)
        )
        Q1 = self.apply_boundary(Q1)

        shock_region_1 = self.discontinuity_mask(Q1)
        Q2 = enforce_physical_state(
            0.75 * self.Q
            + 0.25 * (Q1 + self.dt * self.rhs_euler_hybrid(Q1, shock_region_1))
        )
        Q2 = self.apply_boundary(Q2)

        shock_region_2 = self.discontinuity_mask(Q2)
        self.Q = enforce_physical_state(
            (1.0 / 3.0) * self.Q
            + (2.0 / 3.0)
            * (Q2 + self.dt * self.rhs_euler_hybrid(Q2, shock_region_2))
        )
        self.Q = self.apply_boundary(self.Q)

        if (step_index + 1) % hyperviscosity_interval == 0:
            self.Q = self.apply_hyperviscosity_to_conserved(
                self.Q, self.discontinuity_mask(self.Q)
            )

    def run(self, output_times):
        history_Q = []
        history_time = []
        next_output = 0
        current_time = 0.0

        while next_output < len(output_times) and output_times[next_output] <= 0.0:
            history_Q.append(np.copy(self.Q))
            history_time.append(0.0)
            next_output += 1

        print(
            f"{self.label}: nx={self.nx}, dx={self.dx:.5f}, "
            f"dt={self.dt:.5e}, steps={self.num_steps}"
        )

        for step_index in range(self.num_steps):
            self.step(step_index)
            current_time = (step_index + 1) * self.dt

            while (
                next_output < len(output_times)
                and current_time + 0.5 * self.dt >= output_times[next_output]
            ):
                history_Q.append(np.copy(self.Q))
                history_time.append(output_times[next_output])
                next_output += 1

        while next_output < len(output_times):
            history_Q.append(np.copy(self.Q))
            history_time.append(t_end)
            next_output += 1

        return np.array(history_time), history_Q


def weno5_js_left(v0, v1, v2, v3, v4):
    eps = 1e-6
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


def weno5_js_right(v0, v1, v2, v3, v4):
    eps = 1e-6
    beta0 = (13.0 / 12.0) * (v4 - 2.0 * v3 + v2) ** 2
    beta0 += 0.25 * (v4 - 4.0 * v3 + 3.0 * v2) ** 2
    beta1 = (13.0 / 12.0) * (v3 - 2.0 * v2 + v1) ** 2
    beta1 += 0.25 * (v3 - v1) ** 2
    beta2 = (13.0 / 12.0) * (v2 - 2.0 * v1 + v0) ** 2
    beta2 += 0.25 * (3.0 * v2 - 4.0 * v1 + v0) ** 2

    a0 = 0.1 / (eps + beta0) ** 2
    a1 = 0.6 / (eps + beta1) ** 2
    a2 = 0.3 / (eps + beta2) ** 2
    asum = a0 + a1 + a2

    p0 = (-v4 + 5.0 * v3 + 2.0 * v2) / 6.0
    p1 = (2.0 * v3 + 5.0 * v2 - v1) / 6.0
    p2 = (11.0 * v2 - 7.0 * v1 + 2.0 * v0) / 6.0
    return (a0 * p0 + a1 * p1 + a2 * p2) / asum


def weno5_js_reconstruct_flux(fp, fm):
    ng = 3
    n = fp.size
    fp_pad = np.pad(fp, (ng, ng), mode="edge")
    fm_pad = np.pad(fm, (ng, ng), mode="edge")

    v0 = fp_pad[0 : n + 1]
    v1 = fp_pad[1 : n + 2]
    v2 = fp_pad[2 : n + 3]
    v3 = fp_pad[3 : n + 4]
    v4 = fp_pad[4 : n + 5]
    flux_plus = weno5_js_left(v0, v1, v2, v3, v4)

    v0 = fm_pad[1 : n + 2]
    v1 = fm_pad[2 : n + 3]
    v2 = fm_pad[3 : n + 4]
    v3 = fm_pad[4 : n + 5]
    v4 = fm_pad[5 : n + 6]
    flux_minus = weno5_js_left(v4, v3, v2, v1, v0)

    return flux_plus + flux_minus


class WENO5ReferenceSolver:
    def __init__(self, nx_cells, label):
        self.label = label
        self.dx = (physical_max - physical_min) / nx_cells
        self.nx = nx_cells
        self.x = physical_min + self.dx * np.arange(self.nx)

        rho, u, P = shu_osher_initial_condition(self.x)
        self.Q = conservative_from_primitive(rho, u, P)

        rho0, u0, P0 = primitive_from_conservative(self.Q)
        max_wave_speed = np.max(np.abs(u0) + np.sqrt(gamma * P0 / rho0))
        self.dt = cfl * self.dx / max_wave_speed
        self.num_steps = int(np.ceil(t_end / self.dt))
        self.dt = t_end / self.num_steps

    def rhs(self, Q_curr):
        Q_safe = enforce_physical_state(Q_curr)
        rho_arr, u_arr, P_arr = primitive_from_conservative(Q_safe)
        F_arr = euler_flux(Q_safe)
        alpha = float(np.max(np.abs(u_arr) + np.sqrt(gamma * P_arr / rho_arr)))

        flux_half = np.zeros((3, self.nx + 1))
        for k in range(3):
            fp = 0.5 * (F_arr[k] + alpha * Q_safe[k])
            fm = 0.5 * (F_arr[k] - alpha * Q_safe[k])
            flux_half[k] = weno5_js_reconstruct_flux(fp, fm)

        return -(flux_half[:, 1:] - flux_half[:, :-1]) / self.dx

    def step(self):
        Q1 = enforce_physical_state(self.Q + self.dt * self.rhs(self.Q))
        Q2 = enforce_physical_state(0.75 * self.Q + 0.25 * (Q1 + self.dt * self.rhs(Q1)))
        self.Q = enforce_physical_state(
            (1.0 / 3.0) * self.Q + (2.0 / 3.0) * (Q2 + self.dt * self.rhs(Q2))
        )

    def run(self, output_times):
        history_Q = []
        history_time = []
        next_output = 0
        current_time = 0.0

        print(
            f"{self.label}: nx={self.nx}, dx={self.dx:.5f}, "
            f"dt={self.dt:.5e}, steps={self.num_steps}"
        )

        for step_index in range(self.num_steps):
            self.step()
            current_time = (step_index + 1) * self.dt

            while (
                next_output < len(output_times)
                and current_time + 0.5 * self.dt >= output_times[next_output]
            ):
                history_Q.append(np.copy(self.Q))
                history_time.append(output_times[next_output])
                next_output += 1

        while next_output < len(output_times):
            history_Q.append(np.copy(self.Q))
            history_time.append(t_end)
            next_output += 1

        return np.array(history_time), history_Q


def physical_slice(solver, Q_arr):
    mask = (solver.x >= physical_min) & (solver.x <= physical_max)
    x_plot = solver.x[mask]
    rho_arr, u_arr, P_arr = primitive_from_conservative(Q_arr[:, mask])
    entropy_arr = entropy_function(rho_arr, P_arr)
    return x_plot, rho_arr, P_arr, u_arr, entropy_arr


def compute_final_solutions():
    output_times = np.array([t_end])

    print("Computing high-resolution WENO-JS reference solution...")
    reference_solver = WENO5ReferenceSolver(reference_nx_physical, "Reference WENO-JS")
    _reference_times, reference_history = reference_solver.run(output_times)

    print("Computing validation solution...")
    numerical_solver = HybridEulerSolver(nx_physical, "Hybrid")
    _numerical_times, numerical_history = numerical_solver.run(output_times)

    return (
        numerical_solver,
        numerical_history,
        reference_solver,
        reference_history,
    )


def main():
    (
        numerical_solver,
        numerical_history,
        reference_solver,
        reference_history,
    ) = compute_final_solutions()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    titles = [
        "Pressure",
        "Density",
        "Velocity",
        "Density Zoom",
    ]
    colors = ["tab:blue", "tab:blue", "tab:blue", "tab:blue"]
    ylims = [
        (0.5, 11.5),
        (0.6, 4.8),
        (-0.15, 2.9),
        (3.0, 5.0),
    ]

    x_num, rho_num, P_num, u_num, _entropy_num = physical_slice(
        numerical_solver, numerical_history[-1]
    )
    x_ref, rho_ref, P_ref, u_ref, _entropy_ref = physical_slice(
        reference_solver, reference_history[-1]
    )

    ref_values = [P_ref, rho_ref, u_ref, rho_ref]
    num_values = [P_num, rho_num, u_num, rho_num]
    xlims = [
        (physical_min, physical_max),
        (physical_min, physical_max),
        (physical_min, physical_max),
        (4.5, 8.0),
    ]

    fig.suptitle(
        "Di Renzo 2020 Shu-Osher Shock Tube\n"
        f"Final solution at t = {t_end:.3f} s, N = {nx_physical}",
        fontsize=15,
        fontweight="bold",
    )

    for ax, title, color, ylim, xlim, ref_value, num_value in zip(
        axs.flat, titles, colors, ylims, xlims, ref_values, num_values
    ):
        ax.plot(x_ref, ref_value, "k-", lw=1.2, label="Reference solution")
        ax.plot(
            x_num,
            num_value,
            color=color,
            marker="o",
            ms=3.0,
            lw=1.2,
            label="Hybrid solver",
            zorder=3,
        )

        ax.set_title(title)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best")

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    if os.getenv("SHU_OSHER_SAVE_PDF", "0") == "1":
        output_paths = [
            Path(__file__).with_name("Shu_Osher_t1p8.pdf"),
            Path(os.getenv("TEMP", ".")).joinpath("Shu_Osher_t1p8.pdf"),
        ]
        for output_path in output_paths:
            try:
                fig.savefig(output_path, bbox_inches="tight")
                print(f"Matplotlib backend is non-interactive. Saved figure to: {output_path}")
                break
            except OSError:
                continue

    plt.show(block=True)


if __name__ == "__main__":
    main()
