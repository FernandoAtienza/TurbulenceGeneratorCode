# ============================================================
#  Viscous Burgers: Pure Eq. 39 & Eq. 41 Compact FD (t = 0.5)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
L_min, L_max = -1.0, 1.0
dx = 1/15
nx = 30  # 30 unique intervals for periodic math
dt = 0.001
nu = 0.01 / np.pi
t_end = 0.5
num_steps = int(t_end / dt)

x_solve = np.linspace(L_min, L_max - dx, nx)
x_plot = np.linspace(L_min, L_max, nx + 1)

u = -np.sin(np.pi * x_solve)

# ============================================================
# 1. First Derivative Matrix (Wang Eq. 39 & 40)
# ============================================================
a2 = 4/9; b2 = 1/36
a2_bar = 20/27; b2_bar = 25/216

A1_lil = sp.diags([b2, a2, 1.0, a2, b2], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
A1_lil[0, -1] = a2; A1_lil[0, -2] = b2; A1_lil[1, -1] = b2
A1_lil[-1, 0] = a2; A1_lil[-2, 0] = b2; A1_lil[-1, 1] = b2
solve_A1 = spla.factorized(A1_lil.tocsc())

B1_lil = sp.diags([-b2_bar/dx, -a2_bar/dx, 0.0, a2_bar/dx, b2_bar/dx], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
B1_lil[0, -1] = -a2_bar/dx; B1_lil[0, -2] = -b2_bar/dx; B1_lil[1, -1] = -b2_bar/dx
B1_lil[-1, 0] = a2_bar/dx; B1_lil[-2, 0] = b2_bar/dx; B1_lil[-1, 1] = b2_bar/dx
B1 = B1_lil.tocsc()

def d1_eq39(arr):
    """First spatial derivative using Eq. 39"""
    return solve_A1(B1 @ arr)

# ============================================================
# 2. Second Derivative Matrix (Wang Eq. 41 & 42)
# ============================================================
a3 = 344/1179; b3 = (38*a3 - 9)/214
a3_bar = (696 - 1191*a3)/428; b3_bar = (1227*a3 - 147)/1070

A2_lil = sp.diags([b3, a3, 1.0, a3, b3], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
A2_lil[0, -1] = a3; A2_lil[0, -2] = b3; A2_lil[1, -1] = b3
A2_lil[-1, 0] = a3; A2_lil[-2, 0] = b3; A2_lil[-1, 1] = b3
solve_A2 = spla.factorized(A2_lil.tocsc())

c_mid = -2*(a3_bar + b3_bar)/dx**2
B2_lil = sp.diags([b3_bar/dx**2, a3_bar/dx**2, c_mid, a3_bar/dx**2, b3_bar/dx**2], [-2, -1, 0, 1, 2], shape=(nx, nx)).tolil()
B2_lil[0, -1] = a3_bar/dx**2; B2_lil[0, -2] = b3_bar/dx**2; B2_lil[1, -1] = b3_bar/dx**2
B2_lil[-1, 0] = a3_bar/dx**2; B2_lil[-2, 0] = b3_bar/dx**2; B2_lil[-1, 1] = b3_bar/dx**2
B2 = B2_lil.tocsc()

def d2_eq41(arr):
    """Second spatial derivative using Eq. 41"""
    return solve_A2(B2 @ arr)

# ============================================================
# 3. RK3 Integration Loop
# ============================================================
def RHS(arr):
    F = 0.5 * arr**2
    return -d1_eq39(F) + nu * d2_eq41(arr)

for step in range(num_steps):
    u1 = u + dt * RHS(u)
    u2 = 0.75 * u + 0.25 * (u1 + dt * RHS(u1))
    u  = (1.0/3) * u + (2.0/3) * (u2 + dt * RHS(u2))

u_plot = np.append(u, u[0])

# ============================================================
# 4. Exact Solution & Wang Data
# ============================================================
roots, weights = hermgauss(30)
def exact_solution(x_arr, t):
    u_ex = np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        g = np.sqrt(4 * nu * t) * roots
        y = xi - g
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
        u_ex[i] = np.sum(weights * (-np.sin(np.pi * y)) * f_y) / np.sum(weights * f_y)
    return u_ex

def get_8fd_wang_data():
    return np.array([
        [-1.001728, -0.000453], [-0.934313, 0.044411], [-0.866897, 0.248792],
        [-0.801210, 0.109215], [-0.733794, 0.512991], [-0.666379, 0.149094],
        [-0.600691, 0.792145], [-0.535004, 0.183988], [-0.467589, 1.036405],
        [-0.401901, 0.253776], [-0.334486, 1.220846], [-0.267070, 0.363444],
        [-0.201383, 1.335498], [-0.135696, 0.473112], [-0.066551, 1.430211],
        [-0.000864, -0.000453], [0.064823, -1.421148], [0.133967, -0.469033],
        [0.197926, -1.331420], [0.265341, -0.364350], [0.331029, -1.206798],
        [0.398444, -0.254683], [0.464131, -1.017372], [0.533276, -0.179909],
        [0.597234, -0.778097], [0.662921, -0.135045], [0.732066, -0.508912],
        [0.796024, -0.105136], [0.866897, -0.239728], [0.932584, -0.045317]
    ])

# ============================================================
# 5. Plot & Console Output
# ============================================================
print(f"\n--- Output at t = 0.500 (Pure Eq 39 & 41) ---")
print(f"{'x':>10} | {'u (Numerical)':>20}")
print("-" * 35)
for xi, ui in zip(x_plot, u_plot):
    print(f"{xi:10.5f} | {ui:20.15f}")
print("-" * 35)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_plot, exact_solution(x_plot, t_end), 'b-', lw=2, label='Analytical (Cole-Hopf)', zorder=2)
ax.scatter(x_plot, u_plot, color='r', s=40, label='Eq 39 & 41 Compact FD (No Damping)', zorder=3)
ax.scatter(get_8fd_wang_data()[:,0], get_8fd_wang_data()[:,1], color='black', marker='x', s=60, linewidths=2, zorder=4, label='Wang Paper Data (No Damping)')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("Viscous Burgers: Pure Eq. 39 & Eq. 41 Compact Schemes — t = 0.500")
ax.legend(loc="upper right")
ax.grid(True, linestyle='--', alpha=0.6)

plt.show()