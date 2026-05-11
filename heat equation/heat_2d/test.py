import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Network

# ==========================================================
# 1️⃣ 路径配置
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heat_2d.pth")
FIG_SOL = os.path.join(BASE_DIR, "fig_solution.png")
FIG_ERR = os.path.join(BASE_DIR, "fig_abs_error.png")
FIG_CONTOUR = os.path.join(BASE_DIR, "fig_contour_compare.png")

# ==========================================================
# 2️⃣ 全局绘图风格
# ==========================================================
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "figure.dpi": 300,
})

# ==========================================================
# 4️⃣ 设备
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# 5️⃣ 加载模型
# ==========================================================
model = Network(3, 32, 1, 6, nn.Tanh).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ==========================================================
# 6️⃣ 构建测试网格
# ==========================================================
hx = hy = dt = 0.01

x = torch.arange(0,1+hx,hx)
y = torch.arange(0,2+hy,hy)
t = torch.arange(0,3+dt,dt)

Xg,Yg,Tg = torch.meshgrid(x,y,t,indexing='ij')
X = torch.stack([Xg.flatten(),Yg.flatten(),Tg.flatten()],dim=1).to(device)

# ==========================================================
# 7️⃣ PINN预测
# ==========================================================
with torch.no_grad():
    U_pred = model(X).cpu().numpy()

U_pred = U_pred.reshape(Xg.shape)

# ==========================================================
# 8️⃣ 解析解
# ==========================================================
alpha=0.01
L,H=1.0,2.0

X_np=Xg.numpy()
Y_np=Yg.numpy()
T_np=Tg.numpy()

U_true=(np.sin(np.pi*X_np/L)
       *np.sin(np.pi*Y_np/H)
       *np.exp(-alpha*((np.pi/L)**2+(np.pi/H)**2)*T_np))

# ==========================================================
# 9️⃣ 全局相对L2误差计算
# ==========================================================
abs_err=np.abs(U_pred-U_true)
log_err=np.log10(abs_err+1e-12)

rel_l2=np.sqrt(np.sum((U_pred-U_true)**2)/np.sum(U_true**2))
print(f"\nGlobal Relative L2 Error = {rel_l2:.6e}\n")


# ==========================================================
#  热力图时间
# ==========================================================
times=[0,1,2,3]
idx=[int(tt/dt) for tt in times]


# ==========================================================
# 1️⃣ Solution Heatmaps
# ==========================================================
fig,axes=plt.subplots(1,4,figsize=(12,3),constrained_layout=True)

ims=[]
for i,(ti,tt) in enumerate(zip(idx,times)):
    im=axes[i].imshow(
        U_pred[:,:,ti].T,
        extent=[0,1,0,2],
        origin='lower',
        cmap='viridis'
    )
    axes[i].set_title(f"PINN t={tt}")
    # axes[i].set_aspect('equal')   # ⭐ 子图1:1比例，没用
    ims.append(im)

# ⭐ 统一colorbar（右侧）
cbar=fig.colorbar(ims[-1],ax=axes,shrink=0.85)
cbar.set_label("Temperature")

plt.savefig(FIG_SOL,bbox_inches='tight')
plt.close()


# ==========================================================
# 2️⃣ Absolute Error Heatmaps
# ==========================================================
fig,axes=plt.subplots(1,4,figsize=(12,3),constrained_layout=True)

ims=[]
for i,(ti,tt) in enumerate(zip(idx,times)):
    im=axes[i].imshow(
        abs_err[:,:,ti].T,
        extent=[0,1,0,2],
        origin='lower',
        cmap='magma'
    )
    axes[i].set_title(f"|Error| t={tt}")
    # axes[i].set_aspect('equal')   # ⭐ 子图1:1比例，没用
    ims.append(im)

# ⭐ 统一colorbar
cbar=fig.colorbar(ims[-1],ax=axes,shrink=0.85)
cbar.set_label("Absolute Error")

plt.savefig(FIG_ERR,bbox_inches='tight')
plt.close()


# ==========================================================
# 4️⃣ Contour comparison
# ==========================================================
fig,axes=plt.subplots(1,4,figsize=(12,3))
for i,(ti,tt) in enumerate(zip(idx,times)):
    axes[i].contour(X_np[:,:,ti],Y_np[:,:,ti],
                    U_true[:,:,ti],colors='k')
    axes[i].contour(X_np[:,:,ti],Y_np[:,:,ti],
                    U_pred[:,:,ti],colors='r',linestyles='--')
    axes[i].set_title(f"Contour t={tt}")
plt.tight_layout()
plt.savefig(FIG_CONTOUR)
plt.close()

print("✅ 所有论文级图片已自动保存")
