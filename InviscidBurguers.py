#%%
# ============================================================
#   SECTION 1 — Static analytical solution of inviscid Burgers
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
L = 10
dx = 0.05
x0 = np.arange(0, L, dx)

# Initial condition
def u0(x):
    return np.exp(-(x - 3)**2)

u_initial = u0(x0)

# Times to visualize
times = [0, 0.5, 1.0, 1.5]

plt.figure(figsize=(8,5))

for t in times:
    x = x0 + u_initial * t
    u = u_initial
    plt.plot(x, u, label=f"t = {t}")

plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Analytical Solution of Inviscid Burgers Equation")
plt.legend()
plt.grid()
plt.show()

# %%
# ============================================================
# SECTION 2 — Animated solution using characteristics
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain parameters
L = 10
dx = 0.05
dt = 0.05
T = 36

x0 = np.arange(0, L, dx)

# Initial condition
def u0(x):
    return np.exp(-(x - 3)**2)

u_initial = u0(x0)

# Calculate the shock breaking time Tc 
# Tc = -1 / min{du_0/dx}
du0_dx = np.gradient(u_initial, dx)
min_du0_dx = np.min(du0_dx)
Tc = -1.0 / min_du0_dx

# Setup plot
fig, ax = plt.subplots(figsize=(8,5))
line, = ax.plot([], [], 'b-', lw=2, label='Analytical (Inviscid)')

# Text element to display shock formation warning
shock_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=11, 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

ax.set_xlim(0, L + 2)
ax.set_ylim(0, 1.2)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Inviscid Burgers Equation (Method of Characteristics)")
ax.grid()
ax.legend(loc="upper right")

def init():
    line.set_data([], [])
    shock_text.set_text('')
    return line, shock_text

def update(frame):
    t = frame * dt
    
    # Method of characteristics
    x = x0 + u_initial * t
    u = u_initial

    line.set_data(x, u)
    ax.set_title(f"Inviscid Burgers Equation — t = {t:.2f}")
    
    # Update text when characteristics cross and shock forms
    if t >= Tc:
        shock_text.set_text(f"Shock formed!\nt >= Tc ≈ {Tc:.2f}")
        shock_text.set_color('red')
    else:
        shock_text.set_text(f"Smooth flow\nTc ≈ {Tc:.2f}")
        shock_text.set_color('black')

    return line, shock_text

ani = FuncAnimation(
    fig,
    update,
    frames=T,
    init_func=init,
    interval=80,
    blit=True
)

plt.show()

#%%
# ============================================================
# SECTION 3 — Characteristic lines of Inviscid Burgers equation
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# Domain parameters
L = 10
dx = 0.05
x0 = np.arange(0, L, dx)

# Initial condition
def u0(x):
    return np.exp(-(x - 3)**2)

u_initial = u0(x0)

# Time range
t = np.linspace(0, 2, 200)

plt.figure(figsize=(8,5))

# Plot characteristics from several initial points
for i in range(0, len(x0), 8):
    
    x_char = x0[i] + u_initial[i]*t
    
    plt.plot(t, x_char)

plt.xlabel("t")
plt.ylabel("x")
plt.title("Characteristic Lines for Inviscid Burgers Equation")
plt.grid()

plt.show()

# %%
# ============================================================
# SECTION 5 — Inviscid Burgers: 8th-Order FD vs Analytical
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain parameters
L  = 10.0
nx = 201  # Reduced from 801 so the scatter points are clearly visible
dx = L / (nx - 1)
x  = np.linspace(0, L, nx)

# Time parameters
dt = 0.002  # Small time step for RK3 stability
steps_per_frame = 20
T_frames = 50

# Initial condition
u = np.exp(-(x - 3.0)**2)
u_initial = u.copy()

# Precompute indices for periodic boundary wrap
def roll(u, k):
    return np.roll(u, k)

# 8th‑order first derivative (Central Difference)
def d1(u):
    return (  ( 4/5)*(roll(u,-1)-roll(u,1))
            -( 1/5)*(roll(u,-2)-roll(u,2))
            +( 4/105)*(roll(u,-3)-roll(u,3))
            -( 1/280)*(roll(u,-4)-roll(u,4)) )/dx

# RHS of Inviscid Burgers (nu = 0)
def RHS(u):
    return -u * d1(u)

# Calculate theoretical shock breaking time Tc
du0_dx = np.gradient(u_initial, dx)
Tc = -1.0 / np.min(du0_dx)

# Setup plot
fig, ax = plt.subplots(figsize=(8,5))

# Smooth line for analytical solution
line, = ax.plot([], [], 'b-', lw=2, label='Analytical (Characteristics)')
# Scatter for numerical points
scatter = ax.scatter([], [], color='r', s=15, label='Numerical (8th-Order FD)')

# Text element for warnings
info_text = ax.text(0.05, 0.85, '', transform=ax.transAxes, fontsize=10, 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

ax.set_xlim(0, L + 2)
ax.set_ylim(-0.2, 1.4)
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_title("Inviscid Burgers: 8th-Order FD vs Analytical")
ax.grid()
ax.legend(loc="upper right")

def init():
    line.set_data([], [])
    scatter.set_offsets(np.empty((0, 2)))
    info_text.set_text('')
    return line, scatter, info_text

def update(frame):
    global u
    
    # Time integration for numerical solution using RK3
    for _ in range(steps_per_frame):
        u1 = u + dt*RHS(u)
        u2 = 0.75*u + 0.25*(u1 + dt*RHS(u1))
        u  = (1.0/3)*u + (2.0/3)*(u2 + dt*RHS(u2))

    t_current = frame * dt * steps_per_frame
    
    # Analytical solution using method of characteristics
    x_analytical = x + u_initial * t_current
    line.set_data(x_analytical, u_initial)
    
    # Update numerical scatter plot
    scatter.set_offsets(np.c_[x, u])

    # Update text to monitor shock formation
    if t_current >= Tc:
        info_text.set_text(f"t = {t_current:.3f}\nSHOCK FORMED (t >= {Tc:.2f})\nExpect numerical oscillations!")
        info_text.set_color('red')
    else:
        info_text.set_text(f"t = {t_current:.3f}\nSmooth flow (Tc ≈ {Tc:.2f})")
        info_text.set_color('black')

    return line, scatter, info_text

ani = FuncAnimation(
    fig, 
    update, 
    frames=T_frames, 
    init_func=init, 
    interval=100, 
    blit=True
)

plt.show()
# %%

