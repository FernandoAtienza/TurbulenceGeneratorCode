# ============================================================
# Hybrid Scheme: Wang et al. 2010
# 8th-Order Compact FD + WENO-7 + Eq. 63 Hyperviscosity
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
nx = 30  # 30 unique intervals for periodic math
dx = 1/15
dt = 0.001
nu = 0.01 / np.pi

# Hybrid toggles
use_hyperviscosity = True
mn = 0.01       # Numerical hyperviscosity proposed for the hybrid scheme

# Plot timestep
T_frames = 100 
steps_per_frame = 10

x_solve = np.linspace(L_min, L_max - dx, nx)
x_plot = np.linspace(L_min, L_max, nx + 1)

# Initial condition
u_initial = -np.sin(np.pi * x_solve)
u = u_initial.copy()
current_shock_region = np.zeros_like(u, dtype=bool)

# ============================================================
# 1. Base Advection Matrices (8th-Order Compact)
# ============================================================
alpha1_c = 3.0 / 8.0
A_adv_lil = sp.diags([alpha1_c, 1.0, alpha1_c], [-1, 0, 1], shape=(nx, nx)).tolil()
A_adv_lil[0, -1] = alpha1_c; A_adv_lil[-1, 0] = alpha1_c
solve_A_adv = spla.factorized(A_adv_lil.tocsc())

def d2_6th(arr):
    return ( (1/90)*(np.roll(arr, -3) + np.roll(arr, 3))    # f_j+3 + f_j-3
           - (3/20)*(np.roll(arr, -2) + np.roll(arr, 2))    # f_j+2 + f_j-2
           + (3/2) *(np.roll(arr, -1) + np.roll(arr, 1))    # f_j+1 + f_j-1
           - (49/18)*arr ) / (dx**2)                        # f_j

# ============================================================
# 2. Hyperviscosity Matrices (Wang Eqs. 39-42)
# ============================================================
a2_h = 4/9; b2_h = 1/36
a2_t = 20/27; b2_t = 25/216

A_hyp_lil = sp.diags([b2_h, a2_h, 1.0, a2_h, b2_h], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
A_hyp_lil[0, -1] = a2_h; A_hyp_lil[0, -2] = b2_h; A_hyp_lil[1, -1] = b2_h
A_hyp_lil[-1, 0] = a2_h; A_hyp_lil[-2, 0] = b2_h; A_hyp_lil[-1, 1] = b2_h
A_hyp_csc = A_hyp_lil.tocsc()
solve_A_hyp = spla.factorized(A_hyp_csc)

B_hyp_lil = sp.diags([-b2_t/dx, -a2_t/dx, a2_t/dx, b2_t/dx], [-2, -1, 1, 2], shape=(nx, nx)).tolil()
B_hyp_lil[0, -1] = -a2_t/dx; B_hyp_lil[0, -2] = -b2_t/dx; B_hyp_lil[1, -1] = -b2_t/dx
B_hyp_lil[-1, 0] = a2_t/dx; B_hyp_lil[-2, 0] = b2_t/dx; B_hyp_lil[-1, 1] = b2_t/dx
B_hyp = B_hyp_lil.tocsc()

a3_h = 344/1179; b3_h = (38*a3_h - 9)/214
a3_t = (696 - 1191*a3_h)/428; b3_t = (1227*a3_h - 147)/1070

C_hyp_lil = sp.diags([b3_h, a3_h, 1.0, a3_h, b3_h], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
C_hyp_lil[0, -1] = a3_h; C_hyp_lil[0, -2] = b3_h; C_hyp_lil[1, -1] = b3_h
C_hyp_lil[-1, 0] = a3_h; C_hyp_lil[-2, 0] = b3_h; C_hyp_lil[-1, 1] = b3_h
C_hyp_csc = C_hyp_lil.tocsc()
solve_C_hyp = spla.factorized(C_hyp_csc)

D_hyp_lil = sp.diags(
    [b3_t/dx**2, a3_t/dx**2, -2*(a3_t + b3_t)/dx**2, a3_t/dx**2, b3_t/dx**2],
    [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
D_hyp_lil[0, -1] = a3_t/dx**2; D_hyp_lil[0, -2] = b3_t/dx**2; D_hyp_lil[1, -1] = b3_t/dx**2
D_hyp_lil[-1, 0] = a3_t/dx**2; D_hyp_lil[-2, 0] = b3_t/dx**2; D_hyp_lil[-1, 1] = b3_t/dx**2
D_hyp = D_hyp_lil.tocsc()

# --- Semi-Implicit Solver Setup ---
dt_hyp = 5 * dt
L_imp = C_hyp_csc - dt_hyp * mn * D_hyp
solve_L_hyp = spla.factorized(L_imp)

# ============================================================
# 3. WENO-7 Flux Functions
# ============================================================
def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    eps = 1e-10
    q0 = -(1/4)*v1 + (13/12)*v2 - (23/12)*v3 + (25/12)*v4
    q1 =  (1/12)*v2 - (5/12)*v3 + (13/12)*v4 +  (1/4)*v5
    q2 = -(1/12)*v3 + (7/12)*v4 + (7/12)*v5  - (1/12)*v6
    q3 =  (1/4)*v4  + (13/12)*v5 - (5/12)*v6  + (1/12)*v7
    
    IS0 = (v1 * (544*v1 - 3882*v2 + 4642*v3 - 1854*v4) + v2 * (7043*v2 - 17246*v3 + 7042*v4) + v3 * (11003*v3 - 9402*v4) + 2107 * v4**2)
    IS1 = (v2 * (267*v2 - 1642*v3 + 1602*v4 - 494*v5) + v3 * (2843*v3 - 5966*v4 + 1922*v5) + v4 * (3443*v4 - 2522*v5) + 547 * v5**2)
    IS2 = (v3 * (547*v3 - 2522*v4 + 1922*v5 - 494*v6) + v4 * (3443*v4 - 5966*v5 + 1602*v6) + v5 * (2843*v5 - 1642*v6) + 267 * v6**2)
    IS3 = (v4 * (2107*v4 - 9402*v5 + 7042*v6 - 1854*v7) + v5 * (11003*v5 - 17246*v6 + 4642*v7) + v6 * (7043*v6 - 3882*v7) + 547 * v7**2)
    
    d0, d1, d2, d3 = 1/35, 12/35, 18/35, 4/35
    alpha0, alpha1, alpha2, alpha3 = d0/(eps+IS0)**2, d1/(eps+IS1)**2, d2/(eps+IS2)**2, d3/(eps+IS3)**2
    sum_alpha = alpha0 + alpha1 + alpha2 + alpha3
    w0, w1, w2, w3 = alpha0/sum_alpha, alpha1/sum_alpha, alpha2/sum_alpha, alpha3/sum_alpha
    
    return w0*q0 + w1*q1 + w2*q2 + w3*q3

def get_weno7_interface_flux(arr):
    f = 0.5 * arr**2
    alpha = np.max(np.abs(arr))
    fp = 0.5 * (f + alpha * arr)
    fm = 0.5 * (f - alpha * arr)
    
    fp_half = weno7_flux(np.roll(fp,3), np.roll(fp,2), np.roll(fp,1), fp, np.roll(fp,-1), np.roll(fp,-2), np.roll(fp,-3))
    fm_half = weno7_flux(np.roll(fm,-4), np.roll(fm,-3), np.roll(fm,-2), np.roll(fm,-1), fm, np.roll(fm,1), np.roll(fm,2))
    
    return fp_half + fm_half

# ============================================================
# 4. Operators: Hybrid Advection & Eq 63 Hyperviscosity
# ============================================================
def RHS_hybrid_advection_diffusion(arr, shock_mask):
    f = 0.5 * arr**2
    
    # A. 8th-Order Compact Flux
    a1_c = 25/32; b1_c = 1/20; c1_c = -1/480
    F_comp = (c1_c * (np.roll(f, -3) + np.roll(f, 2)) +
              (b1_c + c1_c) * (np.roll(f, -2) + np.roll(f, 1)) +
              (a1_c + b1_c + c1_c) * (np.roll(f, -1) + f))
              
    # B. 7th-Order WENO Flux
    F_weno_raw = get_weno7_interface_flux(arr)
    F_weno_hat = alpha1_c * np.roll(F_weno_raw, -1) + F_weno_raw + alpha1_c * np.roll(F_weno_raw, 1)
    
    # C. Interface Blending
    is_shock_edge = shock_mask & np.roll(shock_mask, -1)
    is_smooth_edge = (~shock_mask) & (~np.roll(shock_mask, -1))
    is_joint_edge = ~(is_shock_edge | is_smooth_edge)
    
    F_hybrid = np.empty_like(arr)
    F_hybrid[is_smooth_edge] = F_comp[is_smooth_edge]
    F_hybrid[is_shock_edge]  = F_weno_hat[is_shock_edge]
    F_hybrid[is_joint_edge]  = 0.5 * (F_comp[is_joint_edge] + F_weno_hat[is_joint_edge])
    
    # D. Evaluate advection derivative & explicit diffusion
    advection = solve_A_adv((F_hybrid - np.roll(F_hybrid, 1)) / dx)
    diffusion = nu * d2_6th(arr)
                       
    return -advection + diffusion

def apply_semi_implicit_hyperviscosity(u_current, shock_mask):
    """Wang Eq. 63: Solves the hyperviscosity using a stable semi-implicit Euler step."""
    # 1. First derivative u_x
    u_x = solve_A_hyp(B_hyp @ u_current)
    
    # 2. G Flux (Smooth regions)
    # Note: /dx already applies the grid scaling
    G_half = (b2_t * np.roll(u_x, 1) + (a2_t + b2_t) * u_x + 
             (a2_t + b2_t) * np.roll(u_x, -1) + b2_t * np.roll(u_x, -2)) / dx
    
    # 3. H Flux (Shock regions) - Anti-symmetric coefficients fixed
    # Note: /dx**2 already applies the second derivative scaling
    D_half_u = (-b3_t * np.roll(u_current, 1) - (a3_t + b3_t) * u_current + 
               (a3_t + b3_t) * np.roll(u_current, -1) + b3_t * np.roll(u_current, -2)) / (dx**2)
    H_half = A_hyp_csc @ solve_C_hyp(D_half_u)
    
    # 4. Blend G and H
    is_shock_edge = shock_mask & np.roll(shock_mask, -1)
    is_smooth_edge = (~shock_mask) & (~np.roll(shock_mask, -1))
    is_joint_edge = ~(is_shock_edge | is_smooth_edge)
    
    G_mod = np.empty_like(G_half)
    G_mod[is_smooth_edge] = G_half[is_smooth_edge]
    G_mod[is_shock_edge] = H_half[is_shock_edge]
    G_mod[is_joint_edge] = 0.5 * (G_half[is_joint_edge] + H_half[is_joint_edge])
    
    # 5. Take divergence for the explicit RHS term
    # We DO NOT divide by dx here, as G_mod already carries the spatial dimension!
    E = solve_A_hyp(G_mod - np.roll(G_mod, 1))
    
    # 6. Implicit Solve: (C - dt*mn*D) u_new = C * (u_old - dt*mn*E)
    rhs_implicit = u_current - dt_hyp * mn * E
    return solve_L_hyp(C_hyp_csc @ rhs_implicit)

# ------------------------------------------------------------
# Wang 2010 Hybrid Digitized Data
# ------------------------------------------------------------
def get_wang_hybrid_data():
    return np.array([
        [-1.0007757261757766, -0.011070110701107083],
        [-0.9351485800296768,  0.07195571955719582],
        [-0.865664153266412,   0.15498154981549805],
        [-0.8000405654972653,  0.23247232472324675],
        [-0.7344134193511653,  0.31549815498154987],
        [-0.6649325509648538,  0.39298892988929834],
        [-0.599301846441801,   0.48154981549815523],
        [-0.5336853754265602,  0.5479704797047966],
        [-0.46420450704024874, 0.6254612546125462],
        [-0.40051667633359056, 0.6918819188191878],
        [-0.33296800663281445, 0.7638376383763834],
        [-0.2673515356175741,  0.830258302583025],
        [-0.19981709942461012, 0.8800738007380069],
        [-0.13613638547185847, 0.9354243542435423],
        [-0.06865888331014458, 0.8966789667896675],
        [ 0.00019215235546776732, -0.005535055350553542],
        [ 0.06710743095859129, -0.9188191881918822],
        [ 0.1307347692570464,  -0.946494464944649],
        [ 0.19826564707305705, -0.9022140221402217],
        [ 0.26580720001992697, -0.8413284132841334],
        [ 0.33142011265821436, -0.7804428044280441],
        [ 0.39896878235899047, -0.7084870848708482],
        [ 0.46458525337423096, -0.6420664206642065],
        [ 0.5321374814519602,  -0.5645756457564574],
        [ 0.599686151152736,   -0.4926199261992614],
        [ 0.665313297298836,   -0.4095940959409594],
        [ 0.7328655253765655,  -0.3321033210332107],
        [ 0.7984926715226646,  -0.24907749077490737],
        [ 0.8660484579773473,  -0.16605166051660536],
        [ 0.9335971276781236,  -0.09409594095941043]
    ])

# ============================================================
# 5. Exact Solution & Animation
# ============================================================
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

fig, ax = plt.subplots(figsize=(9, 6))
line_ana, = ax.plot(x_plot, exact_solution(x_plot, 0), 'b-', lw=2, label='Analytical (Cole-Hopf)', zorder=2)
scatter_num = ax.scatter([], [], color='g', s=30, zorder=5)

ax.scatter([], [], color='g', s=30, label='8th-Order Compact FD (Smooth)')
ax.scatter([], [], color='r', s=30, label='WENO-7 (Shock Area)')

# Wang et al. (Hybrid) Data point
scatter_wang = ax.scatter([], [], color='black', facecolors='none', marker='s', s=45, linewidths=1.75, zorder=6, label='Wang et al. 2010 (Hybrid)')

ax.set_xlim(L_min, L_max); ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.legend(loc="upper right"); ax.grid(True, linestyle='--', alpha=0.6)

def init():
    global u
    u = u_initial.copy()
    line_ana.set_data(x_plot, exact_solution(x_plot, 0))
    scatter_num.set_offsets(np.c_[x_plot, np.append(u, u[0])])
    scatter_num.set_color('g')
    scatter_wang.set_offsets(np.empty((0, 2)))
    ax.set_title("Hybrid Viscous Burgers — t = 0.000")

def update(frame):
    global u, current_shock_region
    
    for step in range(steps_per_frame):
        # 1. Update Shock Sensor (Dilate by 3 points)
        theta = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        theta_rms = np.sqrt(np.mean(theta**2))
        
        if theta_rms < 1e-12:
            is_shock_node = np.zeros_like(u, dtype=bool)
        else:
            is_shock_node = theta < -3.0 * theta_rms
            
        shock_region = np.copy(is_shock_node)
        for k in range(1, 4):  
            shock_region |= np.roll(is_shock_node, k)
            shock_region |= np.roll(is_shock_node, -k)
            
        current_shock_region[:] = shock_region
        
        # 2. RK3 Integration (Advection + Physical Diffusion)
        u1 = u + dt * RHS_hybrid_advection_diffusion(u, shock_region)
        u2 = 0.75*u + 0.25*(u1 + dt * RHS_hybrid_advection_diffusion(u1, shock_region))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt * RHS_hybrid_advection_diffusion(u2, shock_region))

        # 3. Explicit 5-Step Operator Split for Hyperviscosity
        if use_hyperviscosity and (step + 1) % 5 == 0:
            u = apply_semi_implicit_hyperviscosity(u, shock_region)

    t_current = (frame + 1) * steps_per_frame * dt
    
    # Append the periodic boundary for plotting
    u_plot = np.append(u, u[0])
    shock_plot = np.append(current_shock_region, current_shock_region[0])
    
    scatter_num.set_offsets(np.c_[x_plot, u_plot])
    scatter_num.set_color(np.where(shock_plot, 'r', 'g'))
    line_ana.set_data(x_plot, exact_solution(x_plot, t_current))
    
    # Overlay the Wang Hybrid Data at the final frame
    if frame == T_frames - 1:
        wang_data = get_wang_hybrid_data()
        scatter_wang.set_offsets(wang_data)

    ax.set_title(f"Hybrid Scheme (Compact 8th FD + WENO-7) — t = {t_current:.3f}")

ani = FuncAnimation(
    fig, update, frames=T_frames, init_func=init, interval=30, blit=False, repeat=False 
)
plt.show()