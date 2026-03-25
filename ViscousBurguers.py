# %%
# ============================================================
# SECTION 1 — Viscous Burgers numerical simulation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10
dx = 0.05
dt = 0.01
nu = 0.05
T = 200

x = np.arange(0, L, dx)

# Initial condition
u = np.exp(-(x-3)**2)

# Storage for plotting
solutions = []

for n in range(T):

    un = u.copy()

    for i in range(1,len(x)-1):

        advection = un[i]*(un[i] - un[i-1])/dx
        diffusion = nu*(un[i+1] - 2*un[i] + un[i-1])/dx**2

        u[i] = un[i] - dt*advection + dt*diffusion

    if n % 40 == 0:
        solutions.append(u.copy())

# Plot snapshots
plt.figure(figsize=(8,5))

times = np.arange(0,len(solutions))

for i,sol in enumerate(solutions):
    plt.plot(x,sol,label=f"t={i*40*dt:.2f}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Viscous Burgers Equation")
plt.legend()
plt.grid()

plt.show()

# %%
# ============================================================
# SECTION 2 — Animated viscous Burgers evolution
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
L = 10
dx = 0.05
dt = 0.01
nu = 0.05
T = 300

x = np.arange(0, L, dx)

# Initial condition
u = np.exp(-(x-3)**2)

fig, ax = plt.subplots(figsize=(8,5))
line, = ax.plot(x,u,lw=2)

ax.set_xlim(0,L)
ax.set_ylim(0,1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("Viscous Burgers Equation")
ax.grid()

def update(frame):

    global u

    un = u.copy()

    for i in range(1,len(x)-1):

        advection = un[i]*(un[i] - un[i-1])/dx
        diffusion = nu*(un[i+1] - 2*un[i] + un[i-1])/dx**2

        u[i] = un[i] - dt*advection + dt*diffusion

    line.set_ydata(u)

    ax.set_title(f"Viscous Burgers — t={frame*dt:.2f}")

    return line,

ani = FuncAnimation(fig, update, frames=T, interval=40)

plt.show()

# %%
# ============================================================
# SECTION 3 — Eighth-order FD Burgers solver
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Domain
L  = 10.0
nx = 801
dx = L/(nx-1)
x  = np.linspace(0,L,nx)

# Parameters
nu  = 0.01
dt  = 0.001
nt  = 2000

# Initial condition
u = np.exp(-(x-3.0)**2)

# Precompute indices for periodic boundary wrap
def roll(u, k):
    return np.roll(u, k)

# 8th‑order first derivative
def d1(u):
    return (  ( 4/5)*(roll(u,-1)-roll(u,1))
            -( 1/5)*(roll(u,-2)-roll(u,2))
            +( 4/105)*(roll(u,-3)-roll(u,3))
            -( 1/280)*(roll(u,-4)-roll(u,4)) )/dx

# 8th‑order second derivative
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

# RHS of Burgers
def RHS(u):
    ux  = d1(u)
    uxx = d2(u)
    return -u*ux + nu*uxx

# Time‑march using RK3
for n in range(nt):

    u1 = u + dt*RHS(u)
    u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
    u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

# Plot final solution
plt.figure(figsize=(8,4))
plt.plot(x,u,lw=2)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("8th‑Order FD Burgers at t=%.2f"% (nt*dt))
plt.grid()
plt.show()

# %%
# ============================================================
# SECTION 4 — Animated 8th‑Order Burgers solution with scatter
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Reinitialize
u = np.exp(-(x-3.0)**2)

fig, ax = plt.subplots(figsize=(8,5))

# Smooth line for visualization
line, = ax.plot(x,u,'b',lw=2, label='Line')
# Scatter for numerical points
scatter = ax.scatter(x,u,color='r',s=10, label='Numerical points')

ax.set_xlim(0,L)
ax.set_ylim(0,1.2)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("8th‑Order FD Viscous Burgers")
ax.legend()

def update(frame):

    global u
    for _ in range(10):  # sub-step time looping for smoother anim
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    # Update line
    line.set_ydata(u)
    # Update scatter
    scatter.set_offsets(np.c_[x,u])

    ax.set_title(f"8th‑Order FD Burgers — t = {frame*dt*10:.3f}")
    return line, scatter

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()
# %%