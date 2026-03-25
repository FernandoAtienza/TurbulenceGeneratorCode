# %%
# ============================================================
# SECTION 7 — Hybrid Scheme: 8th-Order Compact FD + WENO-5
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters
L_min, L_max = -1.0, 1.0
nx = 101
dx = (L_max - L_min) / (nx - 1)
dt = 0.001
nu = 0.01 / np.pi

T_frames = 100
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# ------------------------------------------------------------
# 1. Tridiagonal Matrix Setup for Implicit Compact Scheme
# ------------------------------------------------------------
alpha1 = 3.0 / 8.0
# Create the periodic tridiagonal LHS matrix: alpha1 * f'_{j-1} + f'_j + alpha1 * f'_{j+1}
A_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_lil[0, -1] = alpha1  # Periodic wrap-around
A_lil[-1, 0] = alpha1  # Periodic wrap-around
A_csc = A_lil.tocsc()
# Pre-factorize the matrix for ultra-fast solving during the loop
solve_A = spla.factorized(A_csc)

# ------------------------------------------------------------
# 2. WENO-5 Flux Functions
# ------------------------------------------------------------
def weno5_flux(v1, v2, v3, v4, v5):
    eps = 1e-6
    q0 = (1/3)*v1 - (7/6)*v2 + (11/6)*v3
    q1 = -(1/6)*v2 + (5/6)*v3 + (1/3)*v4
    q2 = (1/3)*v3 + (5/6)*v4 - (1/6)*v5
    
    IS0 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - 4*v2 + 3*v3)**2
    IS1 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(v2 - v4)**2
    IS2 = (13/12)*(v3 - 2*v4 + v5)**2 + (1/4)*(3*v3 - 4*v4 + v5)**2
    
    alpha0 = 0.1 / (eps + IS0)**2
    alpha1 = 0.6 / (eps + IS1)**2
    alpha2 = 0.3 / (eps + IS2)**2
    
    sum_alpha = alpha0 + alpha1 + alpha2
    return (alpha0*q0 + alpha1*q1 + alpha2*q2) / sum_alpha

def get_weno_interface_flux(u):
    f = 0.5 * u**2
    alpha = np.max(np.abs(u))
    
    fp = 0.5 * (f + alpha * u)
    fm = 0.5 * (f - alpha * u)
    
    v1_p = np.roll(fp, 2);  v2_p = np.roll(fp, 1);  v3_p = fp
    v4_p = np.roll(fp, -1); v5_p = np.roll(fp, -2)
    fp_half = weno5_flux(v1_p, v2_p, v3_p, v4_p, v5_p)
    
    v1_m = np.roll(fm, -3); v2_m = np.roll(fm, -2); v3_m = np.roll(fm, -1)
    v4_m = fm;              v5_m = np.roll(fm, 1)
    fm_half = weno5_flux(v1_m, v2_m, v3_m, v4_m, v5_m)
    
    return fp_half + fm_half

# ------------------------------------------------------------
# 3. Hybrid RHS Evaluation
# ------------------------------------------------------------
def RHS_hybrid(u):
    global current_shock_region  # Store for coloring the plot
    
    # --- Step A: Shock Detection (Eq. based on local dilatation) ---
    theta = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)  # Velocity gradient
    theta_rms = np.sqrt(np.mean(theta**2))
    is_shock_node = theta < -3.0 * theta_rms
    
    # Dilate the shock region by 3 points in each direction
    shock_region = np.copy(is_shock_node)
    for k in range(1, 4):
        shock_region |= np.roll(is_shock_node, k)
        shock_region |= np.roll(is_shock_node, -k)
        
    current_shock_region = shock_region  # Save for animation colors
    
    # --- Step B: 8th-Order Compact FD Flux (Eq. 25) ---
    f = 0.5 * u**2
    a1_c = 25/32; b1_c = 1/20; c1_c = -1/480
    
    F_comp = (c1_c * (np.roll(f, -3) + np.roll(f, 2)) +
              (b1_c + c1_c) * (np.roll(f, -2) + np.roll(f, 1)) +
              (a1_c + b1_c + c1_c) * (np.roll(f, -1) + f))
              
    # --- Step C: Consistent WENO Flux (Eq. 28) ---
    F_weno_raw = get_weno_interface_flux(u)
    F_weno_hat = alpha1 * np.roll(F_weno_raw, -1) + F_weno_raw + alpha1 * np.roll(F_weno_raw, 1)
    
    # --- Step D: Hybrid Flux Blending (Eq. 30) ---
    F_hybrid = np.zeros_like(u)
    
    # Determine edge types (between j and j+1)
    is_shock_edge = shock_region & np.roll(shock_region, -1)
    is_smooth_edge = (~shock_region) & (~np.roll(shock_region, -1))
    is_joint_edge = ~(is_shock_edge | is_smooth_edge)
    
    F_hybrid[is_smooth_edge] = F_comp[is_smooth_edge]
    F_hybrid[is_shock_edge]  = F_weno_hat[is_shock_edge]
    F_hybrid[is_joint_edge]  = 0.5 * (F_comp[is_joint_edge] + F_weno_hat[is_joint_edge])
    
    # --- Step E: Solve Implicit System for Advection ---
    R = (F_hybrid - np.roll(F_hybrid, 1)) / dx
    advection = solve_A(R)
    
    # --- Step F: Viscous Term (8th-Order Central) ---
    diffusion = nu * ( -(1/560)*np.roll(u,-4) + (8/315)*np.roll(u,-3)
                       - (1/5)*np.roll(u,-2) + (8/5)*np.roll(u,-1)
                       - (205/72)*u
                       + (8/5)*np.roll(u,1) - (1/5)*np.roll(u,2)
                       + (8/315)*np.roll(u,3) - (1/560)*np.roll(u,4) ) / dx**2
                       
    return -advection + diffusion

# ------------------------------------------------------------
# 4. Analytical Solution (Cole-Hopf)
# ------------------------------------------------------------
roots, weights = hermgauss(30)
def exact_solution(x_arr, t):
    if t == 0:
        return -np.sin(np.pi * x_arr)
    u_ex = np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        g = np.sqrt(4 * nu * t) * roots
        y = xi - g
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
        u_ex[i] = np.sum(weights * (-np.sin(np.pi * y)) * f_y) / np.sum(weights * f_y)
    return u_ex

# ------------------------------------------------------------
# 5. Animation Setup
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='g', s=20, zorder=5)

# Dummy scatters just for the legend
ax.scatter([], [], color='g', s=20, label='8th-Order Compact FD (Smooth)')
ax.scatter([], [], color='r', s=20, label='WENO-5 (Shock Area)')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid()

current_shock_region = np.zeros_like(u, dtype=bool)

def init():
    global u
    u = u_initial.copy()
    line_ana.set_data(x, u)
    scatter_num.set_offsets(np.c_[x, u])
    scatter_num.set_color('g')
    ax.set_title("Hybrid Viscous Burgers — t = 0.000")

def update(frame):
    global u
    
    for _ in range(steps_per_frame):
        u1 = u + dt*RHS_hybrid(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS_hybrid(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS_hybrid(u2))

    t_current = (frame + 1) * steps_per_frame * dt
    
    # Update positions
    scatter_num.set_offsets(np.c_[x, u])
    
    # Dynamically update colors based on the shock sensor!
    colors = np.where(current_shock_region, 'r', 'g')
    scatter_num.set_color(colors)
    
    # Update analytical
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    ax.set_title(f"Hybrid Scheme (Compact FD + WENO) — t = {t_current:.3f}")

ani = FuncAnimation(
    fig,
    update,
    frames=T_frames,
    init_func=init,
    interval=30,
    blit=False,
    repeat=False 
)
plt.show()
# %%