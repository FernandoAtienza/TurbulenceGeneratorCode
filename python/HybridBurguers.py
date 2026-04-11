# ============================================================
# Hybrid Scheme: ç
# 8th-Order Compact FD + WENO-7 + Numerical Hyperviscosity
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters
L_min, L_max = -1.0, 1.0
nx = 31
dx = 1/15
dt = 0.001
nu = 0.01 / np.pi

# Numerical hyperviscosity
mn = 0.01
use_flux_hypervisc = True

# Plot timestep
T_frames = 100 # x0.001 sec
steps_per_frame = 10

# Domain
x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# ------------------------------------------------------------
# 1. Tridiagonal Matrix Setup for Implicit Compact Scheme
# ------------------------------------------------------------
alpha1 = 3.0 / 8.0
# Create the periodic tridiagonal LHS matrix: alpha1 * f'_{j-1} + f'_j + alpha1 * f'_{j+1}
# [A]*f' = f
A_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_lil[0, -1] = alpha1  # Periodic wrap-around
A_lil[-1, 0] = alpha1  # Periodic wrap-around
A_csc = A_lil.tocsc()
# Pre-factorize the matrix for fast solving during the loop
solve_A = spla.factorized(A_csc)

# Coefficients
a2 = 4/9
b2 = 1/36

a3 = 344/1179
b3 = (38*a3 - 9)/214

a3_tilde = (696 - 1191*a3)/428
b3_tilde = (1227*a3 - 147)/1070

# Build B matrix 
B_lil = sp.diags(
    [-b2/dx, -a2/dx, a2/dx, b2/dx],
    [-2, -1, 1, 2],
    shape=(nx, nx)
).tolil()

# Periodic wrap
B_lil[0, -1] = -a2/dx
B_lil[0, -2] = -b2/dx
B_lil[1, -1] = -b2/dx

B_lil[-1, 0] = a2/dx
B_lil[-2, 0] = b2/dx
B_lil[-1, 1] = b2/dx

B = B_lil.tocsc()

# Build C matrix
C_lil = sp.diags([b3, a3, 1.0, a3, b3], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()

C_lil[0, -1] = a3
C_lil[0, -2] = b3
C_lil[1, -1] = b3
C_lil[-1, 0] = a3
C_lil[-2, 0] = b3
C_lil[-1, 1] = b3

C = C_lil.tocsc()
solve_C = spla.factorized(C)

# Build D matrix
D_lil = sp.diags(
    [b3_tilde/dx**2, a3_tilde/dx**2, -2*(a3_tilde + b3_tilde)/dx**2,
     a3_tilde/dx**2, b3_tilde/dx**2],
    [-2, -1, 0, 1, 2],
    shape=(nx, nx)
).tolil()

# Periodic wrap
D_lil[0, -1] = a3_tilde/dx**2
D_lil[0, -2] = b3_tilde/dx**2
D_lil[1, -1] = b3_tilde/dx**2

D_lil[-1, 0] = a3_tilde/dx**2
D_lil[-2, 0] = b3_tilde/dx**2
D_lil[-1, 1] = b3_tilde/dx**2

D = D_lil.tocsc()

# ------------------------------------------------------------
# 2. WENO-7 Flux Functions
# ------------------------------------------------------------
def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    # Computes the 7th-order WENO flux interpolation.
    eps = 1e-10
    
    # 3rd-degree polynomial reconstructions (4 sub-stencils)
    q0 = -(1/4)*v1 + (13/12)*v2 - (23/12)*v3 + (25/12)*v4
    q1 =  (1/12)*v2 - (5/12)*v3 + (13/12)*v4 +  (1/4)*v5
    q2 = -(1/12)*v3 + (7/12)*v4 + (7/12)*v5  - (1/12)*v6
    q3 =  (1/4)*v4  + (13/12)*v5 - (5/12)*v6  + (1/12)*v7
    
    # Smoothness indicators (Balsara & Shu, 2000)
    IS0 = (v1 * (544*v1 - 3882*v2 + 4642*v3 - 1854*v4) +
           v2 * (7043*v2 - 17246*v3 + 7042*v4) +
           v3 * (11003*v3 - 9402*v4) +
           2107 * v4**2)

    IS1 = (v2 * (267*v2 - 1642*v3 + 1602*v4 - 494*v5) +
           v3 * (2843*v3 - 5966*v4 + 1922*v5) +
           v4 * (3443*v4 - 2522*v5) +
           547 * v5**2)

    IS2 = (v3 * (547*v3 - 2522*v4 + 1922*v5 - 494*v6) +
           v4 * (3443*v4 - 5966*v5 + 1602*v6) +
           v5 * (2843*v5 - 1642*v6) +
           267 * v6**2)

    IS3 = (v4 * (2107*v4 - 9402*v5 + 7042*v6 - 1854*v7) +
           v5 * (11003*v5 - 17246*v6 + 4642*v7) +
           v6 * (7043*v6 - 3882*v7) +
           547 * v7**2)
    
    # Optimal linear weights for WENO-7
    d0, d1, d2, d3 = 1/35, 12/35, 18/35, 4/35
    
    # Unnormalized non-linear weights
    alpha0 = d0 / (eps + IS0)**2
    alpha1 = d1 / (eps + IS1)**2
    alpha2 = d2 / (eps + IS2)**2
    alpha3 = d3 / (eps + IS3)**2
    
    # Normalized WENO weights
    sum_alpha = alpha0 + alpha1 + alpha2 + alpha3
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha
    w3 = alpha3 / sum_alpha
    
    return w0*q0 + w1*q1 + w2*q2 + w3*q3


def get_weno7_interface_flux(u):
    # Computes the interface fluxes for the WENO-7 scheme.
    f = 0.5 * u**2
    alpha = np.max(np.abs(u)) # Maximum local wave speed
    
    # Lax-Friedrichs Flux Splitting
    fp = 0.5 * (f + alpha * u)  # Positive moving flux
    fm = 0.5 * (f - alpha * u)  # Negative moving flux
    
    # Positive Flux (F^+_{i+1/2})
    v1_p = np.roll(fp, 3)   # i-3
    v2_p = np.roll(fp, 2)   # i-2
    v3_p = np.roll(fp, 1)   # i-1
    v4_p = fp               # i
    v5_p = np.roll(fp, -1)  # i+1
    v6_p = np.roll(fp, -2)  # i+2
    v7_p = np.roll(fp, -3)  # i+3
    fp_half = weno7_flux(v1_p, v2_p, v3_p, v4_p, v5_p, v6_p, v7_p)
    
    # Negative Flux (F^-_{i+1/2})
    v1_m = np.roll(fm, -4)  # i+4
    v2_m = np.roll(fm, -3)  # i+3
    v3_m = np.roll(fm, -2)  # i+2
    v4_m = np.roll(fm, -1)  # i+1
    v5_m = fm               # i
    v6_m = np.roll(fm, 1)   # i-1
    v7_m = np.roll(fm, 2)   # i-2
    fm_half = weno7_flux(v1_m, v2_m, v3_m, v4_m, v5_m, v6_m, v7_m)
    
    return fp_half + fm_half

# ------------------------------------------------------------
# Hyperviscosity - Wang et al. 2010 Implementation
# Equation 55: df/dt = other terms + m_n[f''_n - (f'_n)'_n]
# With shock-switch from Eq. 63
# ------------------------------------------------------------

def hyperviscosity_simplifed(u, shock_region):
    """
    Direct implementation of hyperviscosity.
    Term: m_n[u_xx - (u_x)_x] with shock switching.
    """
    # Compute first derivative u_x using implicit operator
    u_x = solve_A(B @ u)
    
    # Compute second derivative two ways:
    # u_xx: direct second derivative
    u_xx = solve_C(D @ u)
    
    # (u_x)_x: derivative of derivative (via implicit operator)
    u_x_x = solve_A(B @ u_x)
    
    # Base hyperviscosity term (dissipative)
    hypervisc = u_xx - u_x_x
    
    # Apply shock switch: zero out in shock regions
    hypervisc[shock_region] = 0.0
    
    return hypervisc


def hyperviscosity_flux(u, shock_region):
    """
    Flux-conservative formulation with shock switch.
    Implements the decomposition from Eq. 60-63 of Wang et al. 2010.
    Computes: m_n[F''_n - (F'_n)'_n] with shock-aware switching.
    """
    # Coefficients from compact schemes
    a2 = 4/9
    b2 = 1/36
    a3 = 344/1179
    b3 = (38*a3 - 9)/214
    a3_t = (696 - 1191*a3)/428
    b3_t = (1227*a3 - 147)/1070
    
    # Compute first derivative u_x
    u_x = solve_A(B @ u)
    
    # Compute u_xx directly
    u_xx = solve_C(D @ u)
    
    # Compute (u_x)_x using the same implicit compact scheme 
    u_x_x = solve_A(B @ u_x)
    
    # Build the flux G_{j+1/2} = B^{+1/2} * u_x 
    # B^{+1/2} is tetra-diagonal with entries [b2/h, (a2+b2)/h, (a2+b2)/h, b2/h]
    # This amounts to evaluating u_x at the j+1/2 interfaces
    G_half = (
        (b2/dx) * np.roll(u_x, 1) + 
        ((a2 + b2)/dx) * u_x + 
        ((a2 + b2)/dx) * np.roll(u_x, -1) + 
        (b2/dx) * np.roll(u_x, -2)
    )
    
    # Build the flux H_{j+1/2} = A * C^{-1} * D^{+1/2} * u 
    # D^{+1/2} is tetra-diagonal with same structure
    D_half_u = (
        (b3_t/dx**2) * np.roll(u, 1) +
        ((a3_t + b3_t)/dx**2) * u +
        ((a3_t + b3_t)/dx**2) * np.roll(u, -1) +
        (b3_t/dx**2) * np.roll(u, -2)
    )
    # H = A * C^{-1} * D^{+1/2} * u
    H_half = solve_A(A_csc @ solve_C(D_half_u))
    
    # Apply shock switch from Eq. 63
    G_modified = np.copy(G_half)
    
    # Identify edge types: smoothness on both sides vs shock on one/both sides
    smooth_both = (~shock_region) & (~np.roll(shock_region, -1))
    shock_both = shock_region & np.roll(shock_region, -1)
    mixed = ~(smooth_both | shock_both)  # Joint edges
    
    # Apply the switch: use G in smooth regions, H in shock regions, average at joints
    G_modified[smooth_both] = G_half[smooth_both]
    G_modified[shock_both] = H_half[shock_both]
    G_modified[mixed] = 0.5 * (G_half[mixed] + H_half[mixed])
    
    #  Compute flux divergence: (G_{j+1/2} - G_{j-1/2})/dx 
    flux_div = (G_modified - np.roll(G_modified, 1)) / dx
    
    return flux_div

# ------------------------------------------------------------
# 3. Hybrid RHS Evaluation
# ------------------------------------------------------------
def RHS_hybrid(u):
    global current_shock_region  # Store for coloring the plot
    
    # Step A: Shock Detection (based on local dilatation)
    theta = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)  # Velocity gradient
    theta_rms = np.sqrt(np.mean(theta**2))
    # Sensor at velocity gradient x3 the velocity root mean square
    is_shock_node = theta < -3.0 * theta_rms
    
    # Dilate the shock region by 3 points in each direction to get stencil
    shock_region = np.copy(is_shock_node)
    for k in range(1, 4):
        shock_region |= np.roll(is_shock_node, k)
        shock_region |= np.roll(is_shock_node, -k)
        
    current_shock_region = shock_region  # Save for animation colors
    
    # Step B: 8th-Order Compact FD Flux (Eq. 25)
    f = 0.5 * u**2
    a1_c = 25/32; b1_c = 1/20; c1_c = -1/480
    
    F_comp = (c1_c * (np.roll(f, -3) + np.roll(f, 2)) +
              (b1_c + c1_c) * (np.roll(f, -2) + np.roll(f, 1)) +
              (a1_c + b1_c + c1_c) * (np.roll(f, -1) + f))
              
    # Step C: Consistent WENO Flux (Eq. 28)
    # Using the new WENO-7 interface flux calculation
    F_weno_raw = get_weno7_interface_flux(u)
    F_weno_hat = alpha1 * np.roll(F_weno_raw, -1) + F_weno_raw + alpha1 * np.roll(F_weno_raw, 1)
    
    # Step D: Hybrid Flux Blending (Eq. 30)
    F_hybrid = np.zeros_like(u)
    
    # Determine edge types (between j and j+1)
    is_shock_edge = shock_region & np.roll(shock_region, -1)
    is_smooth_edge = (~shock_region) & (~np.roll(shock_region, -1))
    is_joint_edge = ~(is_shock_edge | is_smooth_edge)
    
    F_hybrid[is_smooth_edge] = F_comp[is_smooth_edge]
    F_hybrid[is_shock_edge]  = F_weno_hat[is_shock_edge]
    F_hybrid[is_joint_edge]  = 0.5 * (F_comp[is_joint_edge] + F_weno_hat[is_joint_edge])
    
    # Step E: Solve Implicit System for Advection
    R = (F_hybrid - np.roll(F_hybrid, 1)) / dx
    advection = solve_A(R)
    
    # Step F: Viscous Term (8th-Order Central)
    diffusion = nu * ( -(1/560)*np.roll(u,-4) + (8/315)*np.roll(u,-3)
                       - (1/5)*np.roll(u,-2) + (8/5)*np.roll(u,-1)
                       - (205/72)*u
                       + (8/5)*np.roll(u,1) - (1/5)*np.roll(u,2)
                       + (8/315)*np.roll(u,3) - (1/560)*np.roll(u,4) ) / dx**2

    # Step G: Numerical Hyperviscosity
    # Direct simple formulation from Eq. 55: m_n[u_xx - (u_x)_x] with shock suppression
    # Compute first derivative u_x
    u_x = solve_A(B @ u)
    
    # Compute second derivative two ways and combine
    u_xx = solve_C(D @ u)         # Direct second derivative
    u_x_x = solve_A(B @ u_x)      # Derivative of derivative
    
    # The hyperviscosity term (should be dissipative)
    hypervisc_term = u_xx - u_x_x
    hypervisc_term[shock_region] = 0.0  # Zero out in shock regions
    hypervisc = mn * hypervisc_term
                       
    return -advection + diffusion + hypervisc

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
ax.scatter([], [], color='r', s=20, label='WENO-7 (Shock Area)')

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

    ax.set_title(f"Hybrid Scheme (Compact FD + WENO-7) — t = {t_current:.3f}")

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