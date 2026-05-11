import torch
import numpy as np
import scipy.io as sio
from train import DTYPE,DEVICE
from plot import get_step_id,load_model

if __name__ == "__main__":
    data = sio.loadmat(r"D:\AC.mat")
    uu = data["uu"]        # (512, 201)
    x_ref = data["x"].squeeze()   # (512,)
    tt = data["tt"].squeeze()     # (201,)

    # x 转成 torch tensor
    x_torch = torch.tensor(x_ref, dtype=DTYPE, device=DEVICE).view(-1, 1)

    # ===============================
    # 误差计算
    # ===============================
    rel_errors = []  #(201,)

    u_pred_all = np.zeros_like(uu)  # ⭐保存整个网格的预测

    for j, t_val in enumerate(tt):
        t_val = float(t_val)  # 对201个时间节点,分别计算在一维网格512个x上的相对L2误差

        step_id = get_step_id(t_val)
        model = load_model(step_id)

        # 构造 t tensor
        t_torch = torch.full_like(x_torch, t_val)

        # PINN 预测
        with torch.no_grad():
            u_pred = model(
                torch.cat([x_torch, t_torch], dim=1)
            ).cpu().numpy().squeeze()  # (512,)
        
        u_pred_all[:, j] = u_pred  # ⭐保存每个时间步的预测

        # 参考解
        u_true = uu[:, j]

        # 相对 L2 误差
        rel_l2 = np.linalg.norm(u_pred - u_true) / np.linalg.norm(u_true)
        rel_errors.append(rel_l2)   # list,finally len=201

        if j % 20 == 0:  # 计算t=0,0.2,0.4,0.6,0.8,1.0
            print(f"t = {t_val:.3f}, relative L2 error = {rel_l2:.3e}")

    # ===============================
    # 所有t时刻的L2相对误差中:mean/max
    # ===============================
    mean_rel_error = np.mean(rel_errors)     # 201个时间节点上的平均相对L2误差
    max_rel_error = np.max(rel_errors)       # 201个时间节点上的最大相对L2误差

    print("\n===============================")
    print(f"Average relative L2 error: {mean_rel_error:.3e}")
    print(f"Max relative L2 error:     {max_rel_error:.3e}")
    print("===============================")


    # ===============================
    # ⭐整个 xt(512*201) 网格上的相对 L2
    # ===============================
    rel_l2_xt = np.linalg.norm(u_pred_all - uu) / np.linalg.norm(uu)
    print(f"Relative L2 error on entire xt grid (512x201): {rel_l2_xt:.3e}")


'''
👉该文件用来计算
1、201个时间节点分别的 在一维网格(512个x从-1到1)上的 相对L2误差.打印t-0,2,4,6,8,10的.
2、整个xt(512*201)网格上的相对L2误差
⭐暂定就画一张图,2d和3d的误差目前还没有高精度真解📋
'''


'''
t = 0.000, relative L2 error = 6.734e-03
t = 0.100, relative L2 error = 6.222e-03
t = 0.200, relative L2 error = 7.237e-03
t = 0.300, relative L2 error = 9.468e-03
t = 0.400, relative L2 error = 1.263e-02
t = 0.500, relative L2 error = 1.569e-02
t = 0.600, relative L2 error = 1.849e-02
t = 0.700, relative L2 error = 2.160e-02
t = 0.800, relative L2 error = 2.579e-02
t = 0.900, relative L2 error = 3.114e-02
t = 1.000, relative L2 error = 3.728e-02

===============================
Average relative L2 error: 1.693e-02
Max relative L2 error:     3.728e-02
===============================
Relative L2 error on entire xt grid (512x201): 2.295e-02
'''