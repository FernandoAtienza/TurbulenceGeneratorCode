# ============================================================
#  Animated 8th-Order Compact FD vs Analytical (Section 4.1)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Parameters from Section 4.1
L_min, L_max = -1.0, 1.0
dx = 1/15
nx = 30  # Strictly 30 unique intervals for periodic math
dt = 0.001
nu = 0.01 / np.pi

T_frames = 50  # Ends exactly at t = 0.5
steps_per_frame = 10

# Domain for solver (x=1 is wrapped, so we exclude it here)
x_solve = np.linspace(L_min, L_max - dx, nx)
# Domain for plotting (includes x=1)
x_plot = np.linspace(L_min, L_max, nx + 1)

# Initial condition
u_initial = -np.sin(np.pi * x_solve)
u = u_initial.copy()

# ------------------------------------------------------------
# 1. 8th-Order Compact Scheme for 1st Derivative (Advection)
# ------------------------------------------------------------
alpha1 = 3.0 / 8.0
a1 = 25.0 / 32.0
b1 = 1.0 / 20.0
c1 = -1.0 / 480.0

A_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_lil[0, -1] = alpha1  
A_lil[-1, 0] = alpha1  
solve_A = spla.factorized(A_lil.tocsc())

def d1_compact(arr):
    rhs = ( a1 * (np.roll(arr, -1) - np.roll(arr, 1)) +
            b1 * (np.roll(arr, -2) - np.roll(arr, 2)) +
            c1 * (np.roll(arr, -3) - np.roll(arr, 3)) ) / dx
    return solve_A(rhs)

# ------------------------------------------------------------
# 2. 6th-Order Explicit Scheme for 2nd Derivative (Viscous)
# ------------------------------------------------------------
def d2_6th(arr):
    return ( (1/90)*(np.roll(arr, -3) + np.roll(arr, 3))
           - (3/20)*(np.roll(arr, -2) + np.roll(arr, 2))
           + (3/2) *(np.roll(arr, -1) + np.roll(arr, 1))
           - (49/18)*arr ) / (dx**2)

# ------------------------------------------------------------
# Conservative RHS Formulation
# ------------------------------------------------------------
def RHS(arr):
    # CONSERVATIVE FORMULATION: F = u^2 / 2
    F = 0.5 * arr**2
    Fx = d1_compact(F)
    
    # Viscous term
    uxx = d2_6th(arr)
    
    return -Fx + nu * uxx

# ------------------------------------------------------------
# Exact Solution
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
# Plotting & Wang Data
# ------------------------------------------------------------
def get_8fd_wang_data():
    data2 = np.array([
        [-1.0017286084701815, -0.0004531722054383902],
        [-0.9343128781331029, 0.044410876132930266],
        [-0.8668971477960242, 0.24879154078549814],
        [-0.801210025929127, 0.10921450151057366],
        [-0.7337942955920483, 0.5129909365558907],
        [-0.6663785652549699, 0.14909365558912335],
        [-0.6006914433880728, 0.7921450151057401],
        [-0.5350043215211755, 0.18398791540785475],
        [-0.4675885911840969, 1.036404833836858],
        [-0.4019014693171996, 0.25377643504531666],
        [-0.334485738980121, 1.2208459214501508],
        [-0.2670700086430423, 0.36344410876132893],
        [-0.20138288677614524, 1.3354984894259812],
        [-0.13569576490924795, 0.4731117824773412],
        [-0.06655142610198816, 1.4302114803625374],
        [-0.0008643042350907626, -0.0004531722054383902],
        [0.06482281763180642, -1.4211480362537765],
        [0.13396715643906631, -0.4690332326283988],
        [0.1979256698357823, -1.3314199395770399],
        [0.2653414001728607, -0.36435045317220593],
        [0.3310285220397582, -1.2067975830815716],
        [0.39844425237683656, -0.2546827794561932],
        [0.46413137424373363, -1.01737160120846],
        [0.5332757130509942, -0.17990936555891235],
        [0.5972342264477097, -0.77809667673716],
        [0.6629213483146073, -0.13504531722054436],
        [0.7320656871218669, -0.5089123867069494],
        [0.7960242005185822, -0.10513595166163192],
        [0.8668971477960241, -0.2397280966767379],
        [0.9325842696629216, -0.045317220543807046]
    ])
    return data2

fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x_plot, exact_solution(x_plot, 0), 'b-', lw=2, label='Analytical (Cole-Hopf)', zorder=2)
scatter_num = ax.scatter([], [], color='r', s=25, label='Numerical (8th-Order Compact FD)', zorder=3)
scatter_wang = ax.scatter([], [], color='black', marker='x', s=60, linewidths=2, zorder=4, label='Wang et al. 2010')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.5)

def init():
    global u
    u = u_initial.copy()
    
    # Append first point to end for visual continuity
    u_plot = np.append(u, u[0])
    line_ana.set_data(x_plot, exact_solution(x_plot, 0))
    scatter_num.set_offsets(np.c_[x_plot, u_plot])
    scatter_wang.set_offsets(np.empty((0, 2)))
    ax.set_title("Viscous Burgers: 8th-Order Compact FD — t = 0.000")

def update(frame):
    global u
    
    for _ in range(steps_per_frame):
        # TVD RK3 Integration
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    t_current = (frame + 1) * steps_per_frame * dt
    
    # Append the periodic boundary for plotting
    u_plot = np.append(u, u[0])
    scatter_num.set_offsets(np.c_[x_plot, u_plot])
    
    u_ana = exact_solution(x_plot, t_current)
    line_ana.set_data(x_plot, u_ana)

    if frame == T_frames - 1:
        data_wang = get_8fd_wang_data()
        scatter_wang.set_offsets(data_wang)
        ax.legend(loc="upper right")
        
        # --- NEW: Print the final values to the console ---
        print(f"\n--- Simulated Values at t = {t_current:.3f} ---")
        for xi, ui in zip(x_plot, u_plot):
            print(f"{xi:.16f}, {ui:.16f}")
        print("----------------------------------------\n")

    ax.set_title(f"Viscous Burgers: 8th-Order Compact FD — t = {t_current:.3f}")

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