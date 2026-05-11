import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from train import Network
import os

# =====================================================
# Path
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heat_1d.pth")
PLOT_PATH = os.path.join(BASE_DIR, "plot.png")
MAP_PATH=os.path.join(BASE_DIR, "heatmap.png")

# =====================================================
# Global Plot Style
# =====================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})

# =====================================================
# Device
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================
# Load model（必须和训练完全一致）
# =====================================================
model = Network(
    in_dim=2,
    hidden=20,
    out_dim=1,
    depth=6,
    act=nn.Tanh
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# =====================================================
# Grid
# =====================================================
h = 0.01
k = 0.01

x = torch.arange(0, 1 + h, h)
t = torch.arange(0, 40 + k, k)

X, T = torch.meshgrid(x, t, indexing='ij')
XT = torch.stack([X.flatten(), T.flatten()], dim=1).to(device)

# =====================================================
# Prediction
# =====================================================
with torch.no_grad():
    U_pred = model(XT).reshape(len(x), len(t)).cpu().numpy()

# =====================================================
# True solution
# =====================================================
x_np = x.numpy()
t_np = t.numpy()

X_np, T_np = np.meshgrid(x_np, t_np, indexing='ij')
U_true = np.sin(np.pi * X_np) * np.exp(-0.01 * np.pi**2 * T_np)

abs_error = np.abs(U_pred - U_true)
log_error = np.log10(abs_error + 1e-12)

# 计算并打印相对L2误差
rel_l2_error = np.sqrt(np.sum((U_pred - U_true)**2) / np.sum(U_true**2))
print(f"Relative L2 Error = {rel_l2_error:.6e}")

# =====================================================
# FIGURE 1 — Line plots at selected times
# =====================================================
times = [0, 10, 30, 40]
time_indices = [int(tt / k) for tt in times]

fig, axes = plt.subplots(1, 4, figsize=(16, 3))

for i, (ax, idx, tt) in enumerate(zip(axes, time_indices, times)):
    ax.plot(x_np, U_pred[:, idx], linewidth=2, label="PINN")
    ax.plot(x_np, U_true[:, idx], linestyle='--', linewidth=2, label="True")

    ax.set_title(f"({chr(97+i)}) t = {tt}")
    ax.set_xlabel("x")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)  # 稳定y轴范围
    ax.grid(alpha=0.3)

    if i == 0:
        ax.set_ylabel("u(x,t)")
        ax.legend()

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.show()

# =====================================================
# FIGURE 2 — Heatmap + error
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PINN solution
im0 = axes[0].imshow(
    U_pred,
    extent=[t_np.min(), t_np.max(), x_np.min(), x_np.max()],
    origin='lower',
    aspect='auto',
    cmap="viridis"
)
axes[0].contour(
    T_np,
    X_np,
    U_pred,
    levels=15,
    colors='white',
    linewidths=0.5,
    alpha=0.7
)
axes[0].set_title("(a) PINN Solution")
axes[0].set_xlabel("t")
axes[0].set_ylabel("x")
cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
cbar0.set_label("u(x,t)")

# log10 absolute error
im1 = axes[1].imshow(
    log_error,
    extent=[t_np.min(), t_np.max(), x_np.min(), x_np.max()],
    origin='lower',
    aspect='auto',
    cmap="magma"
)
axes[1].set_title("(b) log10 Absolute Error")
axes[1].set_xlabel("t")
axes[1].set_ylabel("x")
cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
cbar1.set_label("log10|u_pred - u_true|")

plt.tight_layout()
plt.savefig(MAP_PATH, dpi=300)
plt.show()


