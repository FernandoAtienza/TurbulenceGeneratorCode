import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ------------------------------------------------------------
# Parameters from Wang 2010, Section 4.1
# ------------------------------------------------------------
nx = 30  # 30 equal intervals in periodic domain
dx = 1/15
dt = 0.001
nu = 0.01 / np.pi
target_t = 0.5

# Domain (Periodic: we only solve 30 unique points, x=1 is wrapped to x=-1)
x = np.linspace(-1.0, 1.0 - dx, nx)
u = -np.sin(np.pi * x)

# ------------------------------------------------------------
# 1. 8th-Order Compact Scheme for 1st Derivative (Advection)
# ------------------------------------------------------------
# Coefficients from Eq (23)
alpha1 = 3.0 / 8.0
a1 = 25.0 / 32.0
b1 = 1.0 / 20.0
c1 = -1.0 / 480.0

# LHS Tridiagonal Matrix (Periodic)
A_lil = sp.diags([alpha1, 1.0, alpha1], [-1, 0, 1], shape=(nx, nx)).tolil()
A_lil[0, -1] = alpha1  
A_lil[-1, 0] = alpha1  
solve_A = spla.factorized(A_lil.tocsc())

def d1_compact(arr):
    # RHS of Eq (22)
    rhs = ( a1 * (np.roll(arr, -1) - np.roll(arr, 1)) +
            b1 * (np.roll(arr, -2) - np.roll(arr, 2)) +
            c1 * (np.roll(arr, -3) - np.roll(arr, 3)) ) / dx
    return solve_A(rhs)

# ------------------------------------------------------------
# 2. 6th-Order Explicit Scheme for 2nd Derivative (Viscous)
# ------------------------------------------------------------
def d2_6th(arr):
    # Standard 6th-order central difference for 2nd derivative
    return ( (1/90)*(np.roll(arr, -3) + np.roll(arr, 3))
           - (3/20)*(np.roll(arr, -2) + np.roll(arr, 2))
           + (3/2) *(np.roll(arr, -1) + np.roll(arr, 1))
           - (49/18)*arr ) / (dx**2)

# ------------------------------------------------------------
# RHS and RK3 TVD Time Integration
# ------------------------------------------------------------
def RHS(arr):
    # ------------------------------------------------------------
    # WANG 2010 CONSERVATIVE FORMULATION
    # ------------------------------------------------------------
    # Instead of u * du/dx, we evaluate the derivative of the Flux.
    # F = u^2 / 2
    F = 0.5 * arr**2
    
    # Advective term is the derivative of the flux
    Fx = d1_compact(F)
    
    # Viscous term (6th-order explicit)
    uxx = d2_6th(arr)
    
    return -Fx + nu * uxx

# Run simulation to t = 0.5
steps = int(target_t / dt)
for _ in range(steps):
    # 3rd-Order TVD Runge-Kutta
    u1 = u + dt * RHS(u)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * RHS(u1)
    u  = (1.0/3) * u + (2.0/3) * u2 + (2.0/3) * dt * RHS(u2)

# ------------------------------------------------------------
# Analytical Solution (Cole-Hopf via Gauss-Hermite)
# ------------------------------------------------------------
roots, weights = hermgauss(30)

def exact_solution(x_arr, t):
    u_ex = np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        g = np.sqrt(4 * nu * t) * roots
        y = xi - g
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
        u_ex[i] = np.sum(weights * (-np.sin(np.pi * y)) * f_y) / np.sum(weights * f_y)
    return u_ex

# Prepare arrays for plotting (append the periodic boundary x=1)
x_plot = np.append(x, 1.0)
u_plot = np.append(u, u[0])
x_fine = np.linspace(-1, 1, 500)
u_ana = exact_solution(x_fine, target_t)

# Digitized Wang 2010 Data points (for overlay)
wang_data = np.array([
    [-1.0, 0.006], [-0.933, 0.048], [-0.866, 0.241], [-0.801, 0.111],
    [-0.734, 0.514], [-0.667, 0.145], [-0.600, 0.791], [-0.533, 0.178],
    [-0.468, 1.034], [-0.399, 0.258], [-0.334, 1.219], [-0.267, 0.371],
    [-0.199, 1.349], [-0.132, 0.464], [-0.066, 1.433], [ 0.000, 0.006],
    [ 0.066, -1.420], [ 0.135, -0.455], [ 0.199, -1.336], [ 0.267, -0.363],
    [ 0.335, -1.215], [ 0.401, -0.241], [ 0.466, -1.026], [ 0.535, -0.178],
    [ 0.599, -0.783], [ 0.667, -0.132], [ 0.732, -0.506], [ 0.801, -0.107],
    [ 0.868, -0.237], [ 0.933, -0.040], [ 1.000, 0.006]
])

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(x_fine, u_ana, 'b-', lw=2, label='Analytical Solution')
ax.plot(x_plot, u_plot, 'r-o', markersize=5, label='Your Python Base Code (8th Compact)')
ax.plot(wang_data[:, 0], wang_data[:, 1], 'kx', markersize=10, markeredgewidth=2, label='Wang 2010 Paper Data')

ax.set_title(f"Rigorous Baseline Test: 8th-Order Compact FD (t = {target_t})")
ax.set_xlim(-1, 1)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.5)

plt.show()