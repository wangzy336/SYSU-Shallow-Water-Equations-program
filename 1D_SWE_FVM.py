import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

g = 9.81
L = 10.0
nx = 400
dx = L / nx  # 网格间距：10/200=0.05m
CFL = 0.4    # 稳定性系数
Tfinal = 5.0  # 总模拟时间5s
dt_save = 0.02  # 帧保存间隔
fps_animation = 50  # 动画每秒帧数

x = (np.arange(nx) + 0.5) * dx  # 网格中心坐标

# 初始水深分布
h0 = 1.0
eps = 0.05  # 扰动幅度
x0 = L / 2  # 扰动中心
width = 2.0  # 扰动宽度


h = h0 + eps * ((x >= x0 - width/2) & (x <= x0 + width/2)).astype(float)
# h = h0 + eps * np.exp(-((x - x0)**2) / (2 * 0.5**2))  # sigma=0.5


# 初始流速（静止）和守恒变量
u = np.zeros_like(x)  # 初始流速为0
M = h.copy()          # 质量守恒变量 水深
P = h * u             # 动量守恒变量 初始为0

# 绘制并保存初始时刻的图像
plt.figure(figsize=(9, 5))
plt.plot(x, h, color='b')
plt.ylim(0.97, 1.07)
plt.xlabel("x (m)")
plt.ylabel("h (m)")
plt.title("Initial SWE (t = 0.00 s)")
plt.grid(True)
plt.savefig("中心矩形.png", dpi=100)
plt.close()

def flux(M, P):
    h = M
    u = np.where(h > 1e-12, P / h, 0.0)  # 避免除零
    return P, P * u + 0.5 * g * h**2  # 质量通量、动量通量

def max_wave_speed(M, P):
    h = M
    u = np.where(h > 1e-12, P / h, 0.0)
    c = np.sqrt(g * np.maximum(h, 0.0))  # 重力波速
    return np.max(np.abs(u) + c)  # 最大波速（用于计算dt）

def apply_reflective_bcs(M, P):
    # 反射边界条件（左右边界流速反向，水深不变）
    M_ext = np.empty(nx + 2)
    P_ext = np.empty(nx + 2)
    M_ext[1:-1] = M
    P_ext[1:-1] = P
    M_ext[0], M_ext[-1] = M[0], M[-1]
    P_ext[0], P_ext[-1] = -P[0], -P[-1]
    return M_ext, P_ext

def rusanov_step(M, P, dt):
    # Rusanov格式计算守恒变量的更新
    M_ext, P_ext = apply_reflective_bcs(M, P)
    F1_ext, F2_ext = flux(M_ext, P_ext)
    Ml, Mr = M_ext[:-1], M_ext[1:]
    Pl, Pr = P_ext[:-1], P_ext[1:]
    F1l, F1r = F1_ext[:-1], F1_ext[1:]
    F2l, F2r = F2_ext[:-1], F2_ext[1:]
    ul = np.where(Ml>1e-12, Pl/Ml, 0.0)
    ur = np.where(Mr>1e-12, Pr/Mr, 0.0)
    cl = np.sqrt(g * np.maximum(Ml, 0.0))
    cr = np.sqrt(g * np.maximum(Mr, 0.0))
    alpha = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
    F1_num = 0.5 * (F1l + F1r) - 0.5 * alpha * (Mr - Ml)
    F2_num = 0.5 * (F2l + F2r) - 0.5 * alpha * (Pr - Pl)
    M_new = M - (dt / dx) * (F1_num[1:] - F1_num[:-1])
    P_new = P - (dt / dx) * (F2_num[1:] - F2_num[:-1])
    return M_new, P_new

t = 0.0
frames = []  # 存储每帧的水深数据
times = []   # 存储每帧的时间
next_save_time = 0.0

while t < Tfinal:
    maxspeed = max_wave_speed(M, P)
    dt = CFL * dx / maxspeed  # 自适应时间步长（满足CFL条件）
    if t + dt > Tfinal:
        dt = Tfinal - t  # 最后一步确保不超过总时间
    M, P = rusanov_step(M, P, dt)
    t += dt

    # 按dt_save间隔保存帧
    while t >= next_save_time:
        frames.append(M.copy())
        times.append(t)
        next_save_time += dt_save
        if next_save_time > Tfinal:
            break

fig, ax = plt.subplots(figsize=(9, 5))
line, = ax.plot(x, frames[0], color='b')
ax.set_ylim(0.97, 1.07) 
ax.set_xlabel("x (m)")
ax.set_ylabel("h (m)")
ax.set_title(f"SWE (t = {times[0]:.2f} s)")
ax.grid(True)

def update(frame_idx):
    if frame_idx < len(frames) and frame_idx < len(times):
        line.set_ydata(frames[frame_idx]) 
        current_time = times[frame_idx]
        ax.set_title(f"SWE (t = {current_time:.2f} s)") 
    return line,

# 创建动画
ani = FuncAnimation(
    fig, 
    update, 
    frames=np.arange(len(times)),
    interval=1000 / fps_animation, 
    blit=False  
)

# 保存为GIF
ani.save(
    "中心矩形.gif",
    writer="pillow",
    fps=fps_animation, 
    dpi=100 
)

plt.show()