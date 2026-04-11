# ============================================================
#  Animated 8th-Order FD vs Analytical (Section 4.1)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.polynomial.hermite import hermgauss

# Parameters from Section 4.1
L_min, L_max = -1.0, 1.0
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

def get_8fd_wang_data():
    data2 = np.array([
        [-1,                     0.0062937062937062915],
        [-0.9330909090909092,    0.04825174825174816],
        [-0.8661818181818182,    0.24125874125874125],
        [-0.8007272727272727,    0.11118881118881108],
        [-0.7338181818181819,    0.5139860139860142],
        [-0.6669090909090909,    0.1447552447552447],
        [-0.6000000000000001,    0.790909090909091],
        [-0.5330909090909091,    0.17832167832167833],
        [-0.46763636363636374,   1.0342657342657344],
        [-0.3992727272727272,    0.25804195804195795],
        [-0.3338181818181818,    1.2188811188811188],
        [-0.266909090909091,     0.3713286713286714],
        [-0.19854545454545458,   1.348951048951049],
        [-0.13163636363636355,   0.4636363636363636],
        [-0.06618181818181823,   1.4328671328671327],
        [-0.0007272727272726875, 0.0062937062937062915],
        [ 0.06618181818181812,  -1.4202797202797204],
        [ 0.13454545454545452,  -0.45524475524475516],
        [ 0.19854545454545436,  -1.3363636363636364],
        [ 0.266909090909091,    -0.36293706293706296],
        [ 0.33527272727272717,  -1.2146853146853147],
        [ 0.4007272727272726,   -0.24125874125874125],
        [ 0.46618181818181825,  -1.0258741258741257],
        [ 0.5345454545454542,   -0.17832167832167833],
        [ 0.5985454545454547,   -0.7825174825174825],
        [ 0.6669090909090907,   -0.13216783216783212],
        [ 0.7323636363636365,   -0.5055944055944056],
        [ 0.8007272727272723,   -0.10699300699300696],
        [ 0.8676363636363635,   -0.23706293706293713],
        [ 0.9330909090909087,   -0.03986013986013992]
    ])
    return data2

fig, ax = plt.subplots(figsize=(8, 5))

line_ana, = ax.plot(x, u, 'b-', lw=2, label='Analytical (Cole-Hopf)')
scatter_num = ax.scatter(x, u, color='r', s=15, label='Numerical (8th-Order FD)')
scatter_wang = ax.scatter([], [], color='black', marker='x', s=60, linewidths=2, zorder=4, label='Wang et al. 2010')

ax.set_xlim(L_min, L_max)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.legend(loc="upper right")
ax.grid()

def init():
    global u
    u = u_initial.copy()
    line_ana.set_data(x, u)
    scatter_num.set_offsets(np.c_[x, u])
    scatter_wang.set_offsets(np.empty((0, 2)))
    ax.set_title("Viscous Burgers: 8th-Order FD — t = 0.000")

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

    if frame == T_frames - 1:
        data_wang = get_8fd_wang_data()
        scatter_wang.set_offsets(data_wang)
        ax.legend(loc="upper right")

    ax.set_title(f"Viscous Burgers: 8th-Order FD — t = {t_current:.3f}")

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