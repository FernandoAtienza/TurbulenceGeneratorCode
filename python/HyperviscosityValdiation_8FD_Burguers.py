# ============================================================
#  Animated 8th-Order FD: With vs. Without Hyperviscosity
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
nx = 31
dt = 0.001
nu = 0.01 / np.pi

# Hyperviscosity parameter
mn = 0.01

T_frames = 100
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial conditions (tracking two separate simulations)
u_initial = -np.sin(np.pi * x)
u_no_visc = u_initial.copy()
u_visc = u_initial.copy()

# ------------------------------------------------------------
# Matrix Setup for Implicit Hyperviscosity
# ------------------------------------------------------------
alpha1 = 3.0 / 8.0
A_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_lil[0, -1] = alpha1  
A_lil[-1, 0] = alpha1  
A_csc = A_lil.tocsc()
solve_A = spla.factorized(A_csc)

a2 = 4/9; b2 = 1/36
a3 = 344/1179; b3 = (38*a3 - 9)/214
a3_tilde = (696 - 1191*a3)/428; b3_tilde = (1227*a3 - 147)/1070

B_lil = sp.diags([-b2/dx, -a2/dx, a2/dx, b2/dx], [-2, -1, 1, 2], shape=(nx, nx)).tolil()
B_lil[0, -1] = -a2/dx; B_lil[0, -2] = -b2/dx; B_lil[1, -1] = -b2/dx
B_lil[-1, 0] = a2/dx; B_lil[-2, 0] = b2/dx; B_lil[-1, 1] = b2/dx
B = B_lil.tocsc()

C_lil = sp.diags([b3, a3, 1.0, a3, b3], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
C_lil[0, -1] = a3; C_lil[0, -2] = b3; C_lil[1, -1] = b3
C_lil[-1, 0] = a3; C_lil[-2, 0] = b3; C_lil[-1, 1] = b3
C = C_lil.tocsc()
solve_C = spla.factorized(C)

D_lil = sp.diags(
    [b3_tilde/dx**2, a3_tilde/dx**2, -2*(a3_tilde + b3_tilde)/dx**2, a3_tilde/dx**2, b3_tilde/dx**2],
    [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
D_lil[0, -1] = a3_tilde/dx**2; D_lil[0, -2] = b3_tilde/dx**2; D_lil[1, -1] = b3_tilde/dx**2
D_lil[-1, 0] = a3_tilde/dx**2; D_lil[-2, 0] = b3_tilde/dx**2; D_lil[-1, 1] = b3_tilde/dx**2
D = D_lil.tocsc()

# ------------------------------------------------------------
# 8th-Order Explicit Finite Difference Operators
# ------------------------------------------------------------
def roll(arr, k):
    return np.roll(arr, k)

def d1(arr):
    return (  ( 4/5)*(roll(arr,-1)-roll(arr,1))
            -( 1/5)*(roll(arr,-2)-roll(arr,2))
            +( 4/105)*(roll(arr,-3)-roll(arr,3))
            -( 1/280)*(roll(arr,-4)-roll(arr,4)) )/dx

def d2(arr):
    return ( -(1/560)*roll(arr,4) + (8/315)*roll(arr,3) - (1/5)*roll(arr,2) + (8/5)*roll(arr,1)
             - (205/72)*arr + (8/5)*roll(arr,-1) - (1/5)*roll(arr,-2) + (8/315)*roll(arr,-3)
             - (1/560)*roll(arr,-4) )/dx**2

# ------------------------------------------------------------
# RHS Evaluation
# ------------------------------------------------------------
def RHS(arr, apply_hypervisc=False):
    # Base explicit 8th-order terms
    ux  = d1(arr)
    uxx = d2(arr)
    
    if apply_hypervisc:
        # Numerical Hyperviscosity m_n[u_xx - (u_x)_x]
        arr_x = solve_A(B @ arr)
        arr_xx_compact = solve_C(D @ arr)
        arr_x_x = solve_A(B @ arr_x)
        
        hypervisc = mn * (arr_xx_compact - arr_x_x)
        return -arr*ux + nu*uxx + hypervisc
        
    return -arr*ux + nu*uxx

# ------------------------------------------------------------
# Analytical Solution 
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
fig, ax = plt.subplots(figsize=(9, 6))

line_ana, = ax.plot(x, u_initial, 'b-', lw=2, label='Analytical (Cole-Hopf)', zorder=2)
scatter_no_visc = ax.scatter(x, u_no_visc, color='red', marker='x', s=40, label='Without Hyperviscosity', zorder=4)
scatter_visc = ax.scatter(x, u_visc, color='green', marker='o', s=30, label=f'With Hyperviscosity ($m_n = {mn}$)', zorder=3)

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid(True, linestyle='--', alpha=0.7)

def init():
    global u_no_visc, u_visc
    u_no_visc = u_initial.copy()
    u_visc = u_initial.copy()
    line_ana.set_data(x, u_no_visc)
    scatter_no_visc.set_offsets(np.c_[x, u_no_visc])
    scatter_visc.set_offsets(np.c_[x, u_visc])
    ax.set_title("Hyperviscosity Impact Validation — t = 0.000")

def update(frame):
    global u_no_visc, u_visc
    
    for _ in range(steps_per_frame):
        # RK3 for scheme WITHOUT hyperviscosity
        u1_nv = u_no_visc + dt*RHS(u_no_visc, apply_hypervisc=False)
        u2_nv = 0.75*u_no_visc + 0.25*(u1_nv + dt*RHS(u1_nv, apply_hypervisc=False))
        u_no_visc = (1.0/3)*u_no_visc + (2.0/3)*(u2_nv + dt*RHS(u2_nv, apply_hypervisc=False))

        # RK3 for scheme WITH hyperviscosity
        u1_v = u_visc + dt*RHS(u_visc, apply_hypervisc=True)
        u2_v = 0.75*u_visc + 0.25*(u1_v + dt*RHS(u1_v, apply_hypervisc=True))
        u_visc = (1.0/3)*u_visc + (2.0/3)*(u2_v + dt*RHS(u2_v, apply_hypervisc=True))

    t_current = (frame + 1) * steps_per_frame * dt
    
    # Update plot offsets
    scatter_no_visc.set_offsets(np.c_[x, u_no_visc])
    scatter_visc.set_offsets(np.c_[x, u_visc])
    
    # Update analytical solution
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    ax.set_title(f"Hyperviscosity Impact Validation — t = {t_current:.3f}")

ani = FuncAnimation(
    fig, update, frames=T_frames, init_func=init,
    interval=30, blit=False, repeat=False 
)

plt.show()