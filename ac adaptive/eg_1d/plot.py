import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from train import Network,N_NEURONS,N_LAYERS,DEVICE,DT,DTYPE,MODEL_FOLDER

import os
# 当前脚本所在文件夹
PLOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# 两个工具函数
# =====================================================
def get_step_id(t):  # step_id from 1 to 10
    if t==1.0:  # 加入这个逻辑
        return int(10)
    else:
        return int(t // DT) + 1
    # t∈[0,0.1),id=1
    # t∈[0.1,0.2),id=2
    # ...
    # t∈[0.9,1.0),id=10

def load_model(step_id):
    model = Network(2, N_NEURONS, 1, N_LAYERS).to(DEVICE)
    path = os.path.join(MODEL_FOLDER, f"step_{step_id:03d}.pth")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

if __name__ == "__main__":
    # ===============================
    # 读取 .mat 文件
    # ===============================
    data = sio.loadmat(r"D:\AC.mat")

    uu = data["uu"]        # (512, 201)
    x_ref = data["x"].squeeze() # (512,)
    tt = data["tt"].squeeze()   # (201,)
    # x 转 torch
    x_torch = torch.tensor(x_ref, dtype=DTYPE, device=DEVICE).view(-1, 1)


    # ===============================
    # 画图
    # ===============================
    t_plot_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, t_val in zip(axes, t_plot_list):

        # 找到最接近的参考解时间索引
        j = np.argmin(np.abs(tt - t_val))  # t_val=0.2时,index j=40
        t_val = float(tt[j])  # 用参考解中的精确时间

        # 选择模型
        step_id = get_step_id(t_val)
        model = load_model(step_id)

        # PINN 预测
        t_torch = torch.full_like(x_torch, t_val)   # tensor(512,1):values all =t_val
        with torch.no_grad():
            u_pred = model(
                torch.cat([x_torch, t_torch], dim=1)  # tensor(512,2):是mat文件中对应的x
            ).cpu().numpy().squeeze()

        # 参考解
        u_true = uu[:, j]

        # 排序（x不是随机的，不排序应该也行）
        idx = np.argsort(x_ref)
        x_sorted = x_ref[idx]
        u_true_sorted = u_true[idx]
        u_pred_sorted = u_pred[idx]

        # 画图
        ax.plot(x_sorted, u_true_sorted, 'b-', label="Reference")
        ax.plot(x_sorted, u_pred_sorted, 'r--', label="PINN")

        ax.set_title(f"t = {t_val:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.grid(True)

    # 统一图例
    axes[0].legend(loc="best")
    plt.tight_layout()

    fig_filename = os.path.join(PLOT_FOLDER, "AC_PINN_plot_2.png")
    plt.savefig(fig_filename, dpi=300, bbox_inches='tight')
    print(f"图片已保存到 {fig_filename}")




'''
👉该文件用来画出所有需要的图
高精度数值解存放在:"D:\AC.mat"
模型保存在 1D_10_subModel 文件夹下
'''

'''
👉AC.mat 文件格式:
tt : 1*201 double
x : 1*512 double
uu : 512*201 double

tt : index=0,t=0; index=20,t=0.1; ...; index=200,t=1.0
'''
