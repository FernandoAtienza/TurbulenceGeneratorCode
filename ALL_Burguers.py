# %%
# ============================================================
# MASTER SECTION — Analytical vs Upwind vs 8th-Order FD vs WENO-5
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# ------------------------------------------------------------
# 1. Global Parameters
# ------------------------------------------------------------
L_min, L_max = -1.0, 1.0
#nx = 101
#dx = (L_max - L_min) / (nx - 1)
dx = 1/15                      # Grid spacing                          
#nx = (L_max - L_min)/dx + 1    # Number of grid points 
nx = 31
dt = 0.001
nu = 0.01 / np.pi

T_frames = 50
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)

# ------------------------------------------------------------
# 2. Analytical Solution (Cole-Hopf)
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
# 3. Scheme: 8th-Order Central FD
# ------------------------------------------------------------
def roll(u, k):
    return np.roll(u, k)

def d1_8th(u):
    return (  ( 4/5)*(roll(u,-1)-roll(u,1))
            -( 1/5)*(roll(u,-2)-roll(u,2))
            +( 4/105)*(roll(u,-3)-roll(u,3))
            -( 1/280)*(roll(u,-4)-roll(u,4)) )/dx

def d2_8th(u):
    return ( -(1/560)*roll(u,4) + (8/315)*roll(u,3)
             - (1/5)*roll(u,2) + (8/5)*roll(u,1)
             - (205/72)*u
             + (8/5)*roll(u,-1) - (1/5)*roll(u,-2)
             + (8/315)*roll(u,-3) - (1/560)*roll(u,-4) )/dx**2

def RHS_fd8(u):
    return -u * d1_8th(u) + nu * d2_8th(u)

# ------------------------------------------------------------
# 4. Scheme: WENO-5
# ------------------------------------------------------------
def weno5_flux(v1, v2, v3, v4, v5):
    eps = 1e-6
    q0 = (1/3)*v1 - (7/6)*v2 + (11/6)*v3
    q1 = -(1/6)*v2 + (5/6)*v3 + (1/3)*v4
    q2 = (1/3)*v3 + (5/6)*v4 - (1/6)*v5
    
    IS0 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - 4*v2 + 3*v3)**2
    IS1 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(v2 - v4)**2
    IS2 = (13/12)*(v3 - 2*v4 + v5)**2 + (1/4)*(3*v3 - 4*v4 + v5)**2
    
    d0, d1, d2 = 0.1, 0.6, 0.3
    alpha0 = d0 / (eps + IS0)**2
    alpha1 = d1 / (eps + IS1)**2
    alpha2 = d2 / (eps + IS2)**2
    
    sum_alpha = alpha0 + alpha1 + alpha2
    return (alpha0*q0 + alpha1*q1 + alpha2*q2) / sum_alpha

def get_weno_derivative(u):
    f = 0.5 * u**2
    alpha = np.max(np.abs(u))
    fp = 0.5 * (f + alpha * u)
    fm = 0.5 * (f - alpha * u)
    
    # Positive flux
    v1_p = roll(fp, 2); v2_p = roll(fp, 1); v3_p = fp
    v4_p = roll(fp, -1); v5_p = roll(fp, -2)
    fp_half = weno5_flux(v1_p, v2_p, v3_p, v4_p, v5_p)
    
    # Negative flux
    v1_m = roll(fm, -3); v2_m = roll(fm, -2); v3_m = roll(fm, -1)
    v4_m = fm; v5_m = roll(fm, 1)
    fm_half = weno5_flux(v1_m, v2_m, v3_m, v4_m, v5_m)
    
    F_half = fp_half + fm_half
    return (F_half - roll(F_half, 1)) / dx

def RHS_weno(u):
    return -get_weno_derivative(u) + nu * d2_8th(u)

# ------------------------------------------------------------
# 5. Time Integration Helper
# ------------------------------------------------------------
def rk3_step(u, RHS_func):
    """3rd-Order Runge-Kutta time step"""
    u1 = u + dt * RHS_func(u)
    u2 = 0.75*u + 0.25*(u1 + dt * RHS_func(u1))
    return (1.0/3)*u + (2.0/3)*(u2 + dt * RHS_func(u2))

# ------------------------------------------------------------
# 6. Animation and Plotting Setup
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Define the plot elements
line_ana, = ax.plot([], [], 'k-', lw=2.5, zorder=0, label='Analytical (Exact)')
scatter_upw = ax.scatter([], [], color='orange', marker='s', s=20, alpha=0.7, zorder=1, label='1st-Order Upwind FD')
scatter_fd8 = ax.scatter([], [], color='green', marker='^', s=25, alpha=0.8, zorder=2, label='8th-Order Central FD')
scatter_weno = ax.scatter([], [], color='red', marker='o', s=15, alpha=0.9, zorder=3, label='5th-Order WENO')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, linestyle='--', alpha=0.6)

# State variables for each scheme
u_upw = np.zeros_like(x)
u_fd8 = np.zeros_like(x)
u_weno = np.zeros_like(x)

def init():
    global u_upw, u_fd8, u_weno
    u_upw[:]  = u_initial.copy()
    u_fd8[:]  = u_initial.copy()
    u_weno[:] = u_initial.copy()
    
    line_ana.set_data(x, u_initial)
    scatter_upw.set_offsets(np.c_[x, u_upw])
    scatter_fd8.set_offsets(np.c_[x, u_fd8])
    scatter_weno.set_offsets(np.c_[x, u_weno])
    ax.set_title("Master Comparison: Viscous Burgers Equation — t = 0.000")

def update(frame):
    global u_upw, u_fd8, u_weno
    
    for _ in range(steps_per_frame):
        # 1. Update 1st-Order Upwind (Vectorized Explicit Euler for speed)
        adv_p = u_upw * (u_upw - roll(u_upw, 1)) / dx
        adv_m = u_upw * (roll(u_upw, -1) - u_upw) / dx
        advection = np.where(u_upw > 0, adv_p, adv_m)
        diffusion = nu * (roll(u_upw, -1) - 2*u_upw + roll(u_upw, 1)) / dx**2
        u_upw = u_upw - dt * advection + dt * diffusion
        
        # 2. Update 8th-Order FD (RK3)
        u_fd8 = rk3_step(u_fd8, RHS_fd8)
        
        # 3. Update WENO-5 (RK3)
        u_weno = rk3_step(u_weno, RHS_weno)

    # Current time
    t_current = (frame + 1) * steps_per_frame * dt
    
    # Analytical
    u_ana = exact_solution(x, t_current)
    
    # Update plot data
    line_ana.set_data(x, u_ana)
    scatter_upw.set_offsets(np.c_[x, u_upw])
    scatter_fd8.set_offsets(np.c_[x, u_fd8])
    scatter_weno.set_offsets(np.c_[x, u_weno])

    ax.set_title(f"Master Comparison: Viscous Burgers Equation — t = {t_current:.3f}")

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