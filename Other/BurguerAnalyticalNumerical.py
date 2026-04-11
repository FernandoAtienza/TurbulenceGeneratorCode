# %%
# ============================================================
# SECTION 1 — Viscous Burgers numerical vs analytical
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermgauss

# Parameters from Section 4.1 of the paper
L_min, L_max = -1.0, 1.0
nx = 101
dx = (L_max - L_min) / (nx - 1)
dt = 0.001
nu = 0.01 / np.pi
T_steps = 100  # Simulate up to t=0.5

x = np.linspace(L_min, L_max, nx)

# Initial condition: Standing shock wave
u = -np.sin(np.pi * x)

# Storage for plotting
solutions = []
times_to_plot = []

for n in range(T_steps + 1):
    un = u.copy()
    
    # Upwind scheme (backward for u > 0, forward for u < 0) for stability
    for i in range(1, len(x) - 1):
        if un[i] > 0:
            advection = un[i] * (un[i] - un[i-1]) / dx
        else:
            advection = un[i] * (un[i+1] - un[i]) / dx
            
        diffusion = nu * (un[i+1] - 2*un[i] + un[i-1]) / dx**2
        u[i] = un[i] - dt * advection + dt * diffusion

    if n % 100 == 0:
        solutions.append(u.copy())
        times_to_plot.append(n * dt)

# Analytical Solution setup using Gauss-Hermite (30 terms)
roots, weights = hermgauss(30)

def exact_solution(x_arr, t):
    if t == 0:
        return -np.sin(np.pi * x_arr)
    
    u_ex = np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        g = np.sqrt(4 * nu * t) * roots
        y = xi - g
        # f(y) function from the paper's analytical formulation
        f_y = np.exp(-np.cos(np.pi * y) / (2 * np.pi * nu))
        
        numerator = np.sum(weights * (-np.sin(np.pi * y)) * f_y)
        denominator = np.sum(weights * f_y)
        u_ex[i] = numerator / denominator
    return u_ex

# Plot snapshots
plt.figure(figsize=(10, 6))

for sol, t in zip(solutions, times_to_plot):
    # Numerical as scatter
    p = plt.scatter(x, sol, s=15, label=f"Num t={t:.1f}")
    
    # Analytical as solid line
    u_ana = exact_solution(x, t)
    plt.plot(x, u_ana, color=p.get_facecolor()[0], lw=1.5, label=f"Ana t={t:.1f}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Viscous Burgers: Numerical vs Analytical (Section 4.1)")
plt.legend()
plt.grid()
plt.show()

# %%
# ============================================================
# SECTION 2 — Animated viscous Burgers evolution Central Scheme
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# Parameters from Section 4.1
L_min, L_max = -1.0, 1.0
nx = 101
dx = (L_max - L_min) / (nx - 1)
dt = 0.001
nu = 0.01 / np.pi

T_frames = 300         
steps_per_frame = 10   

x = np.linspace(L_min, L_max, nx)

# Store the base initial condition so we can reset it later
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# Gauss-Hermite roots and weights for analytical solution
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

fig, ax = plt.subplots(figsize=(8, 5))

# Plot lines and scatter
line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='r', s=15, label='Numerical (Upwind FD)')

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
    ax.set_title("Viscous Burgers Equation — t = 0.000")
    # No need to return anything when blit=False

def update(frame):
    global u
    
    for _ in range(steps_per_frame):
        un = u.copy()
        for i in range(1, len(x) - 1):
            if un[i] > 0:
                advection = un[i] * (un[i] - un[i-1]) / dx
            else:
                advection = un[i] * (un[i+1] - un[i]) / dx
                
            diffusion = nu * (un[i+1] - 2*un[i] + un[i-1]) / dx**2
            u[i] = un[i] - dt * advection + dt * diffusion

    t_current = (frame + 1) * steps_per_frame * dt
    
    scatter_num.set_offsets(np.c_[x, u])
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    ax.set_title(f"Viscous Burgers Equation — t = {t_current:.3f}")

# Changed blit=False
ani = FuncAnimation(fig, update, frames=T_frames, init_func=init, interval=30, blit=False)

plt.show()

# %%
# ============================================================
# SECTION 3 — Animated 8th-Order FD vs Analytical (Section 4.1)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# Parameters from Section 4.1
L_min, L_max = -1.0, 1.0
#nx = 101
#dx = (L_max - L_min) / (nx - 1)
dx = 1/15
nx = 31
dt = 0.001
nu = 0.01 / np.pi

T_frames = 100
steps_per_frame = 10

x = np.linspace(L_min, L_max, nx)

# Initial condition
u_initial = -np.sin(np.pi * x)
u = u_initial.copy()

# Precompute indices for periodic boundary wrap
def roll(u, k):
    return np.roll(u, k)

def d1(u):
    return (  ( 4/5)*(roll(u,-1)-roll(u,1))
            -( 1/5)*(roll(u,-2)-roll(u,2))
            +( 4/105)*(roll(u,-3)-roll(u,3))
            -( 1/280)*(roll(u,-4)-roll(u,4)) )/dx

def d2(u):
    return ( -(1/560)*roll(u,4)
             + (8/315)*roll(u,3)
             - (1/5)*roll(u,2)
             + (8/5)*roll(u,1)
             - (205/72)*u
             + (8/5)*roll(u,-1)
             - (1/5)*roll(u,-2)
             + (8/315)*roll(u,-3)
             - (1/560)*roll(u,-4) )/dx**2

def RHS(u):
    ux  = d1(u)
    uxx = d2(u)
    return -u*ux + nu*uxx

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

fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='r', s=15, label='Numerical (8th-Order FD)')

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
    ax.set_title("Viscous Burgers: 8th-Order FD — t = 0.000")
    # No need to return anything when blit=False

def update(frame):
    global u
    
    for _ in range(steps_per_frame):
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    t_current = (frame + 1) * steps_per_frame * dt
    
    scatter_num.set_offsets(np.c_[x, u])
    u_ana = exact_solution(x, t_current)
    line_ana.set_data(x, u_ana)

    # Title will now update correctly!
    ax.set_title(f"Viscous Burgers: 8th-Order FD — t = {t_current:.3f}")

# Changed blit=False
ani = FuncAnimation(fig, update, frames=T_frames, init_func=init, interval=30, blit=False, repeat=False )

plt.show()
# %%