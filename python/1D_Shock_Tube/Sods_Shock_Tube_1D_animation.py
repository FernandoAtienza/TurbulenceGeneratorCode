# ============================================================
# 1D Euler Equations: Sod's Shock Tube (ANIMATED)
# Hybrid Scheme: 8th-Order Compact FD + WENO-7
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
gamma = 1.4
R_gas = 1.0
rho_floor = 1e-10
P_floor = 1e-10

# The compact operators are periodic. Use a wider padded domain so the
# artificial periodic discontinuity is far from the validation window [0, 1].
L_min, L_max = -1.0, 2.0
nx = 300
dx = (L_max - L_min) / nx
dt = 0.0005
t_end = 0.2
num_steps = int(t_end / dt)

# Animation Settings
frames = 100
steps_per_frame = max(1, num_steps // frames)

mn = 0.02  # Hyperviscosity parameter
hyperviscosity_interval = 5
sensor_width = 4
compression_threshold = 2.5
jump_threshold = 0.035
x_solve = np.linspace(L_min, L_max - dx, nx)

# Initial conditions (Sod)
rho = np.where(x_solve < 0.5, 1.0, 0.125)
u = np.zeros_like(x_solve)
P = np.where(x_solve < 0.5, 1.0, 0.1)

# Conservative variables Q
Q = np.zeros((3, nx))
Q[0] = rho
Q[1] = rho * u
Q[2] = P / (gamma - 1) + 0.5 * rho * u**2

# ============================================================
# Matrices (8th-Order Compact & Hyperviscosity)
# ============================================================
alpha1_c = 3.0 / 8.0
A_adv_lil = sp.diags([alpha1_c, 1.0, alpha1_c], [-1, 0, 1], shape=(nx, nx)).tolil()
A_adv_lil[0, -1] = alpha1_c; A_adv_lil[-1, 0] = alpha1_c
solve_A_adv = spla.factorized(A_adv_lil.tocsc())

a2_t = 20/27; b2_t = 25/216
a3_h = 344/1179; b3_h = (38*a3_h - 9)/214
a3_t = (696 - 1191*a3_h)/428; b3_t = (1227*a3_h - 147)/1070

A_hyp_lil = sp.diags([1/36, 4/9, 1.0, 4/9, 1/36], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
A_hyp_lil[0, -1] = 4/9; A_hyp_lil[0, -2] = 1/36; A_hyp_lil[1, -1] = 1/36
A_hyp_lil[-1, 0] = 4/9; A_hyp_lil[-2, 0] = 1/36; A_hyp_lil[-1, 1] = 1/36
A_hyp_csc = A_hyp_lil.tocsc()
solve_A_hyp = spla.factorized(A_hyp_csc)

B_hyp_lil = sp.diags([-b2_t/dx, -a2_t/dx, a2_t/dx, b2_t/dx], [-2, -1, 1, 2], shape=(nx, nx)).tolil()
B_hyp_lil[0, -1] = -a2_t/dx; B_hyp_lil[0, -2] = -b2_t/dx; B_hyp_lil[1, -1] = -b2_t/dx
B_hyp_lil[-1, 0] = a2_t/dx; B_hyp_lil[-2, 0] = b2_t/dx; B_hyp_lil[-1, 1] = b2_t/dx
B_hyp = B_hyp_lil.tocsc()

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

dt_hyp = 5 * dt
solve_L_hyp = spla.factorized(C_hyp_csc - dt_hyp * mn * D_hyp)

# ============================================================
# Primitive/Conservative Helpers and Discontinuity Sensor
# ============================================================
def primitive_from_conservative(Q_arr):
    rho_arr = np.maximum(Q_arr[0], rho_floor)
    u_arr = Q_arr[1] / rho_arr
    E_arr = Q_arr[2]
    P_arr = (gamma - 1.0) * (E_arr - 0.5 * rho_arr * u_arr**2)
    P_arr = np.maximum(P_arr, P_floor)
    return rho_arr, u_arr, P_arr

def conservative_from_primitive(rho_arr, u_arr, P_arr):
    Q_arr = np.zeros((3, rho_arr.size))
    rho_arr = np.maximum(rho_arr, rho_floor)
    P_arr = np.maximum(P_arr, P_floor)
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

def relative_jump_sensor(q_arr):
    eps = 1e-14 + 1e-12 * np.max(np.abs(q_arr))
    jump_r = np.abs(np.roll(q_arr, -1) - q_arr) / (np.abs(np.roll(q_arr, -1)) + np.abs(q_arr) + eps)
    jump_l = np.abs(q_arr - np.roll(q_arr, 1)) / (np.abs(q_arr) + np.abs(np.roll(q_arr, 1)) + eps)
    curvature = np.abs(np.roll(q_arr, -1) - 2.0 * q_arr + np.roll(q_arr, 1))
    curvature /= np.abs(np.roll(q_arr, -1)) + 2.0 * np.abs(q_arr) + np.abs(np.roll(q_arr, 1)) + eps
    return np.maximum.reduce([jump_l, jump_r, curvature])

def dilate_periodic_mask(mask, width):
    expanded = mask.copy()
    for k in range(1, width + 1):
        expanded |= np.roll(mask, k)
        expanded |= np.roll(mask, -k)
    return expanded

def discontinuity_mask(Q_arr):
    rho_arr, u_arr, P_arr = primitive_from_conservative(Q_arr)
    theta = (np.roll(u_arr, -1) - np.roll(u_arr, 1)) / (2.0 * dx)
    theta_rms = np.sqrt(np.mean(theta**2))
    compression = np.zeros_like(theta, dtype=bool)
    if theta_rms > 1e-12:
        compression = theta < -compression_threshold * theta_rms

    density_jump = relative_jump_sensor(rho_arr) > jump_threshold
    pressure_jump = relative_jump_sensor(P_arr) > jump_threshold
    internal_energy = P_arr / (rho_arr * (gamma - 1.0))
    energy_jump = relative_jump_sensor(internal_energy) > jump_threshold

    return dilate_periodic_mask(compression | density_jump | pressure_jump | energy_jump, sensor_width)

def interface_masks(node_mask):
    weno_edge = node_mask | np.roll(node_mask, -1)
    smooth_edge = ~weno_edge
    return weno_edge, smooth_edge

# ============================================================
# WENO-7 Flux Splitting & Operators
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
    
    alpha0 = (1/35)/(eps+IS0)**2; alpha1 = (12/35)/(eps+IS1)**2
    alpha2 = (18/35)/(eps+IS2)**2; alpha3 = (4/35)/(eps+IS3)**2
    sum_alpha = alpha0 + alpha1 + alpha2 + alpha3
    
    return (alpha0*q0 + alpha1*q1 + alpha2*q2 + alpha3*q3) / sum_alpha

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

    R = np.array([
        [1.0, 1.0, 1.0],
        [u_roe - c, u_roe, u_roe + c],
        [H_roe - u_roe * c, kinetic, H_roe + u_roe * c],
    ])

    gm1 = gamma - 1.0
    L = np.array([
        [(gm1 * kinetic + u_roe * c) / (2.0 * c2), -(gm1 * u_roe + c) / (2.0 * c2), gm1 / (2.0 * c2)],
        [1.0 - gm1 * kinetic / c2, gm1 * u_roe / c2, -gm1 / c2],
        [(gm1 * kinetic - u_roe * c) / (2.0 * c2), -(gm1 * u_roe - c) / (2.0 * c2), gm1 / (2.0 * c2)],
    ])
    return L, R

def get_weno7_euler_flux_LLF(Q_arr, F_arr, alpha, required_edges=None):
    F_half = np.zeros_like(Q_arr)
    F_plus = 0.5 * (F_arr + alpha * Q_arr)
    F_minus = 0.5 * (F_arr - alpha * Q_arr)
    if required_edges is None:
        indices = range(nx)
    else:
        indices = np.flatnonzero(dilate_periodic_mask(required_edges, 1))

    for i in indices:
        ip1 = (i + 1) % nx
        L, R = roe_eigenvectors(Q_arr[:, i], Q_arr[:, ip1])

        plus_stencil = np.array([L @ F_plus[:, (i + s) % nx] for s in range(-3, 4)])
        minus_stencil = np.array([L @ F_minus[:, (i + s) % nx] for s in range(4, -3, -1)])

        flux_char = np.empty(3)
        for m in range(3):
            flux_char[m] = (
                weno7_flux(*plus_stencil[:, m])
                + weno7_flux(*minus_stencil[:, m])
            )

        F_half[:, i] = R @ flux_char

    return F_half

def RHS_euler_hybrid(Q_curr, shock_mask):
    Q_safe = enforce_physical_state(Q_curr)
    rho, u, P = primitive_from_conservative(Q_safe)
    E = Q_safe[2]
    
    F_curr = np.zeros_like(Q_safe)
    F_curr[0] = Q_safe[1]; F_curr[1] = rho * u**2 + P; F_curr[2] = (E + P) * u
    
    c = np.sqrt(gamma * P / rho)
    alpha = float(np.max(np.abs(u) + c))
    
    a1_c = 25/32; b1_c = 1/20; c1_c = -1/480
    advection = np.zeros_like(Q_safe)
    weno_edge, smooth_edge = interface_masks(shock_mask)
    
    F_weno_raw = get_weno7_euler_flux_LLF(Q_safe, F_curr, alpha, weno_edge)
    
    for k in range(3):
        f = F_curr[k]
        F_comp = (c1_c * (np.roll(f, -3) + np.roll(f, 2)) +
                  (b1_c + c1_c) * (np.roll(f, -2) + np.roll(f, 1)) +
                  (a1_c + b1_c + c1_c) * (np.roll(f, -1) + f))
        
        F_weno_hat = alpha1_c * np.roll(F_weno_raw[k], -1) + F_weno_raw[k] + alpha1_c * np.roll(F_weno_raw[k], 1)
        
        F_hybrid = np.empty_like(f)
        F_hybrid[smooth_edge] = F_comp[smooth_edge]
        F_hybrid[weno_edge] = F_weno_hat[weno_edge]
        
        advection[k] = solve_A_adv((F_hybrid - np.roll(F_hybrid, 1)) / dx)
                       
    return -advection

def apply_semi_implicit_hyperviscosity(u_current, shock_mask):
    u_x = solve_A_hyp(B_hyp @ u_current)
    G_half = (b2_t * np.roll(u_x, 1) + (a2_t + b2_t) * u_x + (a2_t + b2_t) * np.roll(u_x, -1) + b2_t * np.roll(u_x, -2)) / dx
    D_half_u = (-b3_t * np.roll(u_current, 1) - (a3_t + b3_t) * u_current + (a3_t + b3_t) * np.roll(u_current, -1) + b3_t * np.roll(u_current, -2)) / (dx**2)
    H_half = A_hyp_csc @ solve_C_hyp(D_half_u)
    
    shock_edge, smooth_edge = interface_masks(shock_mask)
    
    G_mod = np.empty_like(G_half)
    G_mod[smooth_edge] = G_half[smooth_edge]
    G_mod[shock_edge] = H_half[shock_edge]
    
    E = solve_A_hyp(G_mod - np.roll(G_mod, 1))
    rhs_implicit = u_current - dt_hyp * mn * E
    return solve_L_hyp(C_hyp_csc @ rhs_implicit)

def apply_hyperviscosity_to_conserved(Q_curr, shock_mask):
    Q_visc = np.empty_like(Q_curr)
    for k in range(3):
        Q_visc[k] = apply_semi_implicit_hyperviscosity(Q_curr[k], shock_mask)
    return enforce_physical_state(Q_visc)

# ============================================================
# Exact Riemann Solver (Dynamic Time)
# ============================================================
def exact_sod_dynamic(x_arr, t):
    rho_ex, P_ex, e_ex, u_ex_arr = np.zeros_like(x_arr), np.zeros_like(x_arr), np.zeros_like(x_arr), np.zeros_like(x_arr)
    
    if t <= 0.0:
        rho_ex = np.where(x_arr < 0.5, 1.0, 0.125)
        P_ex = np.where(x_arr < 0.5, 1.0, 0.1)
        e_ex = P_ex / (rho_ex * (gamma - 1))
        return rho_ex, P_ex, e_ex, np.zeros_like(x_arr)

    P3, u3, rho3, rho4 = 0.30313, 0.92745, 0.42632, 0.26557
    cL, c3, V_shock = 1.18321, 0.99772, 1.75215
    
    head = 0.5 - cL * t
    tail = 0.5 + (u3 - c3) * t
    contact = 0.5 + u3 * t
    shock = 0.5 + V_shock * t
    
    for i, x in enumerate(x_arr):
        if x < head: 
            rho_ex[i], P_ex[i], u_ex = 1.0, 1.0, 0.0
        elif x < tail: 
            u_ex = 2/(gamma+1) * (cL + (x - 0.5)/t)
            c = cL - (gamma-1)/2 * u_ex
            rho_ex[i] = 1.0 * (c/cL)**(2/(gamma-1))
            P_ex[i] = 1.0 * (c/cL)**(2*gamma/(gamma-1))
        elif x < contact: 
            rho_ex[i], P_ex[i], u_ex = rho3, P3, u3
        elif x < shock: 
            rho_ex[i], P_ex[i], u_ex = rho4, P3, u3
        else: 
            rho_ex[i], P_ex[i], u_ex = 0.125, 0.1, 0.0
            
        e_ex[i] = P_ex[i] / (rho_ex[i] * (gamma - 1))
        u_ex_arr[i] = u_ex
        
    return rho_ex, P_ex, e_ex, u_ex_arr

# ============================================================
# Compute and Store Frames
# ============================================================
print("Computing fluid evolution. Please wait...")
history_Q = []
history_time = []

for step in range(num_steps):
    if step % steps_per_frame == 0:
        history_Q.append(np.copy(Q))
        history_time.append(step * dt)
    
    shock_region = discontinuity_mask(Q)
    Q1 = enforce_physical_state(Q + dt * RHS_euler_hybrid(Q, shock_region))

    shock_region_1 = discontinuity_mask(Q1)
    Q2 = enforce_physical_state(0.75*Q + 0.25*(Q1 + dt * RHS_euler_hybrid(Q1, shock_region_1)))

    shock_region_2 = discontinuity_mask(Q2)
    Q = enforce_physical_state((1.0/3)*Q + (2.0/3)*(Q2 + dt * RHS_euler_hybrid(Q2, shock_region_2)))

    if (step + 1) % hyperviscosity_interval == 0:
        Q = apply_hyperviscosity_to_conserved(Q, discontinuity_mask(Q))

history_Q.append(np.copy(Q))  # Append final frame
history_time.append(num_steps * dt)
print("Computation complete. Launching animation...")

# ============================================================
# Plot Setup & Animation
# ============================================================
plot_mask = (x_solve >= 0.0) & (x_solve <= 1.0)
x_plot = x_solve[plot_mask]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Store line and scatter objects
lines_exact = []
scatters_num = []
titles = ["Normalized Density", "Normalized Pressure", "Normalized Velocity", "Internal Energy (Unnormalized)"]
ylims = [(-0.1, 1.1), (-0.1, 1.1), (-0.1, 1.1), (1.5, 3.0)]
colors = ['r', 'g', 'm', 'b']

for i, ax in enumerate(axs.flat):
    line, = ax.plot([], [], 'k-', lw=1.5, label='Exact Analytical')
    scatter = ax.scatter([], [], color=colors[i], s=15, label='Hybrid Scheme')
    
    lines_exact.append(line)
    scatters_num.append(scatter)
    
    ax.set_title(titles[i])
    ax.set_xlim(0, 1)
    ax.set_ylim(ylims[i])
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="lower left")

plt.tight_layout(rect=[0, 0.03, 1, 0.92])

def animate(frame):
    current_time = history_time[frame]
    fig.suptitle(f"Sod's Shock Tube Evolution\nTime = {current_time:.3f} s", fontsize=15, fontweight='bold')
    
    Q_frame = history_Q[frame]
    
    # Calculate Numerical variables (sliced to [0, 1])
    rho_num = Q_frame[0, plot_mask]
    u_num = Q_frame[1, plot_mask] / rho_num
    E_num = Q_frame[2, plot_mask]
    P_num = (gamma - 1) * (E_num - 0.5 * rho_num * u_num**2)
    e_num = P_num / (rho_num * (gamma - 1))
    
    # Normalize
    vals_num = [rho_num/1.0, P_num/1.0, u_num/1.0, e_num]
    
    # Calculate Exact analytical variables
    rho_ex, P_ex, e_ex, u_ex = exact_sod_dynamic(x_plot, current_time)
    vals_ex = [rho_ex/1.0, P_ex/1.0, u_ex/1.0, e_ex]
    
    # Update graphical objects
    for i in range(4):
        lines_exact[i].set_data(x_plot, vals_ex[i])
        scatters_num[i].set_offsets(np.c_[x_plot, vals_num[i]])
        
    return lines_exact + scatters_num

# Run the animation
anim = FuncAnimation(fig, animate, frames=len(history_Q), interval=40, blit=False, repeat=False)

plt.show()
