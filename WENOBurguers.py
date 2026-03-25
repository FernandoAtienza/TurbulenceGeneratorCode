# %%
# ============================================================
# SECTION 6 — Animated WENO-5 vs Analytical (Viscous Burgers)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# Parameters
L_min, L_max = -1.0, 1.0
#nx = 101
#dx = (L_max - L_min) / (nx - 1)
nx = 31
dx = 1/15
dt = 0.001
nu = 0.01 / np.pi

T_frames = 50
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# ------------------------------------------------------------
# 5th-Order WENO Advection Scheme
# ------------------------------------------------------------
def weno5_flux(v1, v2, v3, v4, v5):
    """Computes the 5th-order WENO flux interpolation."""
    eps = 1e-6
    
    # 3rd-order polynomial reconstructions
    q0 = (1/3)*v1 - (7/6)*v2 + (11/6)*v3
    q1 = -(1/6)*v2 + (5/6)*v3 + (1/3)*v4
    q2 = (1/3)*v3 + (5/6)*v4 - (1/6)*v5
    
    # Smoothness indicators (measure of local gradients)
    IS0 = (13/12)*(v1 - 2*v2 + v3)**2 + (1/4)*(v1 - 4*v2 + 3*v3)**2
    IS1 = (13/12)*(v2 - 2*v3 + v4)**2 + (1/4)*(v2 - v4)**2
    IS2 = (13/12)*(v3 - 2*v4 + v5)**2 + (1/4)*(3*v3 - 4*v4 + v5)**2
    
    # Optimal linear weights
    d0, d1, d2 = 0.1, 0.6, 0.3
    
    # Unnormalized non-linear weights
    alpha0 = d0 / (eps + IS0)**2
    alpha1 = d1 / (eps + IS1)**2
    alpha2 = d2 / (eps + IS2)**2
    
    # Normalized WENO weights
    sum_alpha = alpha0 + alpha1 + alpha2
    w0 = alpha0 / sum_alpha
    w1 = alpha1 / sum_alpha
    w2 = alpha2 / sum_alpha
    
    return w0*q0 + w1*q1 + w2*q2

def get_weno_derivative(u):
    """Computes the spatial derivative of the flux using Lax-Friedrichs splitting."""
    f = 0.5 * u**2
    alpha = np.max(np.abs(u)) # Maximum local wave speed
    
    # Lax-Friedrichs Flux Splitting
    fp = 0.5 * (f + alpha * u)  # Positive moving flux
    fm = 0.5 * (f - alpha * u)  # Negative moving flux
    
    # --- Positive Flux (F^+_{i+1/2}) ---
    # Stencil leans to the left
    v1_p = np.roll(fp, 2)   # i-2
    v2_p = np.roll(fp, 1)   # i-1
    v3_p = fp               # i
    v4_p = np.roll(fp, -1)  # i+1
    v5_p = np.roll(fp, -2)  # i+2
    fp_half = weno5_flux(v1_p, v2_p, v3_p, v4_p, v5_p)
    
    # --- Negative Flux (F^-_{i+1/2}) ---
    # Stencil leans to the right (indices mirrored)
    v1_m = np.roll(fm, -3)  # i+3
    v2_m = np.roll(fm, -2)  # i+2
    v3_m = np.roll(fm, -1)  # i+1
    v4_m = fm               # i
    v5_m = np.roll(fm, 1)   # i-1
    fm_half = weno5_flux(v1_m, v2_m, v3_m, v4_m, v5_m)
    
    # Total interface flux F_{i+1/2}
    F_half = fp_half + fm_half
    
    # F_{i-1/2} is just F_{i+1/2} shifted to the right by 1 cell
    F_half_shifted = np.roll(F_half, 1)
    
    # Return df/dx
    return (F_half - F_half_shifted) / dx

# ------------------------------------------------------------
# 8th-Order Central Scheme for Viscous Term
# ------------------------------------------------------------
def d2(u):
    """8th-order second derivative for the diffusion term."""
    return ( -(1/560)*np.roll(u,-4) + (8/315)*np.roll(u,-3)
             - (1/5)*np.roll(u,-2) + (8/5)*np.roll(u,-1)
             - (205/72)*u
             + (8/5)*np.roll(u,1) - (1/5)*np.roll(u,2)
             + (8/315)*np.roll(u,3) - (1/560)*np.roll(u,4) ) / dx**2

def RHS(u):
    advection = get_weno_derivative(u)
    diffusion = nu * d2(u)
    return -advection + diffusion

# ------------------------------------------------------------
# Analytical Solution (Cole-Hopf)
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
# Animation Setup
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='r', s=15, label='Numerical (WENO-5)')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid()

def init():
    global u
    u = u_initial.copy()
    line_ana.set_data(x, u)
    scatter_num.set_offsets(np.c_[x, u])
    ax.set_title("Viscous Burgers: WENO-5 — t = 0.000")

def update(frame):
    global u
    
    # Time integration using RK3
    for _ in range(steps_per_frame):
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    t_current = (frame + 1) * steps_per_frame * dt
    
    scatter_num.set_offsets(np.c_[x, u])
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    ax.set_title(f"Viscous Burgers: WENO-5 — t = {t_current:.3f}")

ani = FuncAnimation(fig, update, frames=T_frames, init_func=init, interval=30, blit=False)
plt.show()

# %%
# ============================================================
# SECTION 8 — Animated WENO-7 vs Analytical (Viscous Burgers)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# Parameters
L_min, L_max = -1.0, 1.0
#nx = 101
#dx = (L_max - L_min) / (nx - 1)
nx = 31
dx = 1/15
dt = 0.001
nu = 0.01 / np.pi

T_frames = 50 # x0.001 sec
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# ------------------------------------------------------------
# 7th-Order WENO Advection Scheme
# ------------------------------------------------------------
def weno7_flux(v1, v2, v3, v4, v5, v6, v7):
    """Computes the 7th-order WENO flux interpolation."""
    eps = 1e-6
    
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

def get_weno7_derivative(u):
    """Computes the spatial derivative of the flux using Lax-Friedrichs splitting."""
    f = 0.5 * u**2
    alpha = np.max(np.abs(u)) # Maximum local wave speed
    
    # Lax-Friedrichs Flux Splitting
    fp = 0.5 * (f + alpha * u)  # Positive moving flux
    fm = 0.5 * (f - alpha * u)  # Negative moving flux
    
    # --- Positive Flux (F^+_{i+1/2}) ---
    # Stencil leans to the left (i-3 to i+3)
    v1_p = np.roll(fp, 3)   # i-3
    v2_p = np.roll(fp, 2)   # i-2
    v3_p = np.roll(fp, 1)   # i-1
    v4_p = fp               # i
    v5_p = np.roll(fp, -1)  # i+1
    v6_p = np.roll(fp, -2)  # i+2
    v7_p = np.roll(fp, -3)  # i+3
    fp_half = weno7_flux(v1_p, v2_p, v3_p, v4_p, v5_p, v6_p, v7_p)
    
    # --- Negative Flux (F^-_{i+1/2}) ---
    # Stencil leans to the right (indices mirrored: i+4 down to i-2)
    v1_m = np.roll(fm, -4)  # i+4
    v2_m = np.roll(fm, -3)  # i+3
    v3_m = np.roll(fm, -2)  # i+2
    v4_m = np.roll(fm, -1)  # i+1
    v5_m = fm               # i
    v6_m = np.roll(fm, 1)   # i-1
    v7_m = np.roll(fm, 2)   # i-2
    fm_half = weno7_flux(v1_m, v2_m, v3_m, v4_m, v5_m, v6_m, v7_m)
    
    # Total interface flux F_{i+1/2}
    F_half = fp_half + fm_half
    
    # F_{i-1/2} is just F_{i+1/2} shifted to the right by 1 cell
    F_half_shifted = np.roll(F_half, 1)
    
    # Return df/dx
    return (F_half - F_half_shifted) / dx

# ------------------------------------------------------------
# 8th-Order Central Scheme for Viscous Term
# ------------------------------------------------------------
def d2(u):
    """8th-order second derivative for the diffusion term."""
    return ( -(1/560)*np.roll(u,-4) + (8/315)*np.roll(u,-3)
             - (1/5)*np.roll(u,-2) + (8/5)*np.roll(u,-1)
             - (205/72)*u
             + (8/5)*np.roll(u,1) - (1/5)*np.roll(u,2)
             + (8/315)*np.roll(u,3) - (1/560)*np.roll(u,4) ) / dx**2

def RHS(u):
    advection = get_weno7_derivative(u)
    diffusion = nu * d2(u)
    return -advection + diffusion

# ------------------------------------------------------------
# Analytical Solution (Cole-Hopf)
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
# Numerical Solution (Wang 2010)
# ------------------------------------------------------------
def get_weno_wang_data():
    data = np.array([
        [-0.9997075704692436, -0.012355914499926568],
        [-0.9327070643573699,  0.07660619479062292],
        [-0.7987060521336227,  0.2397033951566303],
        [-0.6681409895142429,  0.3929159544196337],
        [-0.4654215356974607,  0.6252058687490112],
        [-0.33313838031282916, 0.768533786909011],
        [-0.26613787420095547, 0.8327841237602933],
        [-0.19741951124996604, 0.8871499326297432],
        [-0.13041900513809235, 0.9316309872750179],
        [-0.06685436998854088, 0.9019769508448348],
        [ 0.00014613612333280734, 0.012355857939340575],
        [ 0.0671466422352065, -0.901977007405421],
        [ 0.1341471483470802, -0.9464580620506957],
        [ 0.19427575525021856, -0.8871499891903294],
        [ 0.26814816057082735, -0.8278419163299631],
        [ 0.33343065255949456, -0.7734761074605129],
        [ 0.40214917279457474, -0.694398752394139],
        [ 0.5344321708951156, -0.5510708342341393],
        [ 0.6649973121565407, -0.39785838809230745],
        [ 0.8007163385034946, -0.2397034517172163],
        [ 0.9329994938881261, -0.07660625135120891]
    ])
    return data

# ------------------------------------------------------------
# Animation Setup
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='r', s=15, label='Numerical (WENO-7)')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid()

def init():
    global u
    u = u_initial.copy()
    line_ana.set_data(x, u)
    scatter_num.set_offsets(np.c_[x, u])
    ax.set_title("Viscous Burgers: WENO-7 — t = 0.000")

def update(frame):
    global u
    
    # Time integration using RK3
    for _ in range(steps_per_frame):
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    t_current = (frame + 1) * steps_per_frame * dt
    
    # Update numerical + analytical
    scatter_num.set_offsets(np.c_[x, u])
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    ax.set_title(f"Viscous Burgers: WENO-7 — t = {t_current:.3f}")

    # ------------------------------------------------------------
    # Plot Wang (2010) data at the final frame
    # ------------------------------------------------------------
    if frame == T_frames - 1:
        data = get_weno_wang_data()
        x_wang = data[:, 0]
        u_wang = data[:, 1]

        ax.scatter(x_wang, u_wang, color='k', marker='x', s=60, linewidths=2)
        ax.legend()

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
