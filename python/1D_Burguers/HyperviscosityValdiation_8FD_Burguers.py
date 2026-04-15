# ============================================================
#  Animated 8th-Order Compact FD: With vs. Without Hyperviscosity
#  * Validating against Wang 2010 mn=1.0 data at t=0.5
#  * Implements Exact Semi-Implicit Operator Splitting
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L_min, L_max = -1.0, 1.0
dx = 1/15
nx = 30  # Strictly 30 unique intervals for periodic math
dt = 0.001
nu = 0.01 / np.pi

mn = 1.0 # Hyperviscosity parameter

T_frames = 50  # Ends exactly at t = 0.5
steps_per_frame = 10

x_solve = np.linspace(L_min, L_max - dx, nx)
x_plot = np.linspace(L_min, L_max, nx + 1)

u_initial = -np.sin(np.pi * x_solve)
u_no_visc = u_initial.copy()
u_visc = u_initial.copy()

# ============================================================
# 1. Base Advection Matrices (Eq 22-25)
# ============================================================
alpha1 = 3.0 / 8.0; a1 = 25.0 / 32.0; b1 = 1.0 / 20.0; c1 = -1.0 / 480.0
A_adv_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_adv_lil[0, -1] = alpha1; A_adv_lil[-1, 0] = alpha1  
solve_A_adv = spla.factorized(A_adv_lil.tocsc())

def d1_compact(arr):
    rhs = ( a1 * (np.roll(arr, -1) - np.roll(arr, 1)) +
            b1 * (np.roll(arr, -2) - np.roll(arr, 2)) +
            c1 * (np.roll(arr, -3) - np.roll(arr, 3)) ) / dx
    return solve_A_adv(rhs)

def d2_6th(arr):
    return ( (1/90)*(np.roll(arr, -3) + np.roll(arr, 3))
           - (3/20)*(np.roll(arr, -2) + np.roll(arr, 2))
           + (3/2) *(np.roll(arr, -1) + np.roll(arr, 1))
           - (49/18)*arr ) / (dx**2)

# ============================================================
# 2. Wang's Hyperviscosity Matrices (Eq 39 - 42)
# ============================================================
a2 = 4/9; b2 = 1/36
a2_tilde = 20/27; b2_tilde = 25/216

A_hyp_lil = sp.diags([b2, a2, 1.0, a2, b2], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
A_hyp_lil[0, -1] = a2; A_hyp_lil[0, -2] = b2; A_hyp_lil[1, -1] = b2
A_hyp_lil[-1, 0] = a2; A_hyp_lil[-2, 0] = b2; A_hyp_lil[-1, 1] = b2
solve_A_hyp = spla.factorized(A_hyp_lil.tocsc())

B_hyp_lil = sp.diags([-b2_tilde/dx, -a2_tilde/dx, a2_tilde/dx, b2_tilde/dx], [-2, -1, 1, 2], shape=(nx, nx)).tolil()
B_hyp_lil[0, -1] = -a2_tilde/dx; B_hyp_lil[0, -2] = -b2_tilde/dx; B_hyp_lil[1, -1] = -b2_tilde/dx
B_hyp_lil[-1, 0] = a2_tilde/dx; B_hyp_lil[-2, 0] = b2_tilde/dx; B_hyp_lil[-1, 1] = b2_tilde/dx
B_hyp = B_hyp_lil.tocsc()

a3 = 344/1179; b3 = (38*a3 - 9)/214
a3_tilde = (696 - 1191*a3)/428; b3_tilde = (1227*a3 - 147)/1070

C_hyp_lil = sp.diags([b3, a3, 1.0, a3, b3], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
C_hyp_lil[0, -1] = a3; C_hyp_lil[0, -2] = b3; C_hyp_lil[1, -1] = b3
C_hyp_lil[-1, 0] = a3; C_hyp_lil[-2, 0] = b3; C_hyp_lil[-1, 1] = b3
C_hyp = C_hyp_lil.tocsc()

D_hyp_lil = sp.diags(
    [b3_tilde/dx**2, a3_tilde/dx**2, -2*(a3_tilde + b3_tilde)/dx**2, a3_tilde/dx**2, b3_tilde/dx**2],
    [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
D_hyp_lil[0, -1] = a3_tilde/dx**2; D_hyp_lil[0, -2] = b3_tilde/dx**2; D_hyp_lil[1, -1] = b3_tilde/dx**2
D_hyp_lil[-1, 0] = a3_tilde/dx**2; D_hyp_lil[-2, 0] = b3_tilde/dx**2; D_hyp_lil[-1, 1] = b3_tilde/dx**2
D_hyp = D_hyp_lil.tocsc()

dt_hyp = 5 * dt
L_imp = C_hyp - dt_hyp * mn * D_hyp
solve_L_hyp = spla.factorized(L_imp)

# ------------------------------------------------------------
# RHS Evaluation (Pure Physics)
# ------------------------------------------------------------
def RHS(arr):
    F = 0.5 * arr**2
    return -d1_compact(F) + nu * d2_6th(arr)

# ------------------------------------------------------------
# Analytical Solution & Data
# ------------------------------------------------------------
roots, weights = hermgauss(30)
def exact_solution(x_arr, t):
    if t == 0: return -np.sin(np.pi * x_arr)
    u_ex = np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        g = np.sqrt(4 * nu * t) * roots
        y = xi - g
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
        u_ex[i] = np.sum(weights * (-np.sin(np.pi * y)) * f_y) / np.sum(weights * f_y)
    return u_ex

def get_wang_mn1_data():
    return np.array([
        [-1.005028, -0.002773], [-0.935386, 0.085952], [-0.871544, 0.163586],
        [-0.805766, 0.241220], [-0.738053, 0.318854], [-0.668410, 0.407579],
        [-0.602618, 0.463031], [-0.536848, 0.551756], [-0.469159, 0.668207],
        [-0.403321, 0.651571], [-0.337550, 0.740296], [-0.267983, 0.945471],
        [-0.203970, 0.756932], [-0.136302, 0.906654], [-0.068750, 1.233826],
        [-0.002128, 0.002773], [0.066427, -1.222736], [0.132054, -0.912200],
        [0.199722, -0.762477], [0.263731, -0.945471], [0.331370, -0.751386],
        [0.401005, -0.651571], [0.466844, -0.668207], [0.530660, -0.551756],
        [0.600299, -0.457486], [0.664159, -0.407579], [0.727996, -0.324399],
        [0.797639, -0.235675], [0.863417, -0.158041], [0.933070, -0.085952]
    ])

def get_8fd_wang_data_no_visc():
    return np.array([
        [-1.0017286084701815, -0.0004531722054383902], [-0.9343128781331029, 0.044410876132930266],
        [-0.8668971477960242, 0.24879154078549814], [-0.801210025929127, 0.10921450151057366],
        [-0.7337942955920483, 0.5129909365558907], [-0.6663785652549699, 0.14909365558912335],
        [-0.6006914433880728, 0.7921450151057401], [-0.5350043215211755, 0.18398791540785475],
        [-0.4675885911840969, 1.036404833836858], [-0.4019014693171996, 0.25377643504531666],
        [-0.334485738980121, 1.2208459214501508], [-0.2670700086430423, 0.36344410876132893],
        [-0.20138288677614524, 1.3354984894259812], [-0.13569576490924795, 0.4731117824773412],
        [-0.06655142610198816, 1.4302114803625374], [-0.0008643042350907626, -0.0004531722054383902],
        [0.06482281763180642, -1.4211480362537765], [0.13396715643906631, -0.4690332326283988],
        [0.1979256698357823, -1.3314199395770399], [0.2653414001728607, -0.36435045317220593],
        [0.3310285220397582, -1.2067975830815716], [0.39844425237683656, -0.2546827794561932],
        [0.46413137424373363, -1.01737160120846], [0.5332757130509942, -0.17990936555891235],
        [0.5972342264477097, -0.77809667673716], [0.6629213483146073, -0.13504531722054436],
        [0.7320656871218669, -0.5089123867069494], [0.7960242005185822, -0.10513595166163192],
        [0.8668971477960241, -0.2397280966767379], [0.9325842696629216, -0.045317220543807046]
    ])

# ------------------------------------------------------------
# Animation Setup
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

line_ana, = ax.plot(x_plot, exact_solution(x_plot, 0), 'b-', lw=2, label='Analytical', zorder=2)

# This work
scatter_no_visc = ax.scatter([], [], color='red', marker='o', s=30, label='This work (No Damping)', zorder=4)
scatter_visc = ax.scatter([], [], color='green', marker='o', s=30, label=f'This work ($m_n={mn}$)', zorder=4)

# Wang 2010 Data (Squares, unfilled)
scatter_wang_no_visc = ax.scatter([], [], color='red', marker='s', s=45, facecolors='none', linewidths=1.5, zorder=5, label='Wang 2010 (No Damping)')
scatter_wang_visc = ax.scatter([], [], color='green', marker='s', s=45, facecolors='none', linewidths=1.5, zorder=5, label=f'Wang 2010 ($m_n=1.0$)')

ax.set_xlim(L_min, L_max); ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.legend(loc="upper right"); ax.grid(True, linestyle='--', alpha=0.7)

def init():
    global u_no_visc, u_visc
    u_no_visc = u_initial.copy(); u_visc = u_initial.copy()
    line_ana.set_data(x_plot, exact_solution(x_plot, 0))
    scatter_no_visc.set_offsets(np.c_[x_plot, np.append(u_no_visc, u_no_visc[0])])
    scatter_visc.set_offsets(np.c_[x_plot, np.append(u_visc, u_visc[0])])
    scatter_wang_no_visc.set_offsets(np.empty((0, 2)))
    scatter_wang_visc.set_offsets(np.empty((0, 2)))
    ax.set_title("Hyperviscosity Validation — t = 0.000")

def update(frame):
    global u_no_visc, u_visc
    
    for step in range(steps_per_frame):
        # 1. Standard RK3 for the UN-DAMPED equation
        u1_nv = u_no_visc + dt*RHS(u_no_visc)
        u2_nv = 0.75*u_no_visc + 0.25*(u1_nv + dt*RHS(u1_nv))
        u_no_visc = (1.0/3)*u_no_visc + (2.0/3)*(u2_nv + dt*RHS(u2_nv))

        # 2. Standard RK3 for the DAMPED equation (Advection & Physics only)
        u1_v = u_visc + dt*RHS(u_visc)
        u2_v = 0.75*u_visc + 0.25*(u1_v + dt*RHS(u1_v))
        u_visc = (1.0/3)*u_visc + (2.0/3)*(u2_v + dt*RHS(u2_v))

        # 3. WANG'S 5-STEP SEMI-IMPLICIT HYPERVISCOSITY FILTER
        if (step + 1) % 5 == 0:
            ux = solve_A_hyp(B_hyp @ u_visc)
            uxx_explicit = solve_A_hyp(B_hyp @ ux)
            rhs_implicit = u_visc - dt_hyp * mn * uxx_explicit
            u_visc = solve_L_hyp(C_hyp @ rhs_implicit)

    t_current = (frame + 1) * steps_per_frame * dt
    
    u_nv_plot = np.append(u_no_visc, u_no_visc[0])
    u_v_plot = np.append(u_visc, u_visc[0])
    
    scatter_no_visc.set_offsets(np.c_[x_plot, u_nv_plot])
    scatter_visc.set_offsets(np.c_[x_plot, u_v_plot])
    line_ana.set_data(x_plot, exact_solution(x_plot, t_current))

    if frame == T_frames - 1:
        # Map both Wang datasets at the final frame
        scatter_wang_no_visc.set_offsets(get_8fd_wang_data_no_visc())
        scatter_wang_visc.set_offsets(get_wang_mn1_data())
        ax.legend(loc="upper right")

    ax.set_title(f"Hyperviscosity Validation — t = {t_current:.3f}")

ani = FuncAnimation(
    fig, update, frames=T_frames, init_func=init, interval=30, blit=False, repeat=False 
)
plt.show()