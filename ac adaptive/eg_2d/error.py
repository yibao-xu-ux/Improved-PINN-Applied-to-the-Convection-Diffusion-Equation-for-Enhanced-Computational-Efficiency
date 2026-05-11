import os

import numpy as np
import torch

from plot import REFERENCE_PATH, evaluate_pinn, get_step_id, load_model_2d


def main():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"{REFERENCE_PATH} not found. Run generate_reference.py first."
        )

    data = np.load(REFERENCE_PATH)
    x = data["x"]
    y = data["y"]
    tt = data["tt"]
    uu = data["uu"]
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    rel_errors = []
    sq_error_sum = 0.0
    sq_ref_sum = 0.0
    model_cache = {}

    for j, t_val in enumerate(tt):
        t_val = float(t_val)
        step_id = get_step_id(t_val)
        if step_id not in model_cache:
            model_cache[step_id] = load_model_2d(step_id)

        u_pred = evaluate_pinn(model_cache[step_id], x_grid, y_grid, t_val)
        u_ref = uu[j]
        diff = u_pred - u_ref

        rel_l2 = np.linalg.norm(diff) / np.linalg.norm(u_ref)
        rel_errors.append(rel_l2)
        sq_error_sum += float(np.sum(diff**2))
        sq_ref_sum += float(np.sum(u_ref**2))

        if j == 0 or j == len(tt) - 1 or np.isclose(t_val % 1.0, 0.0, atol=1e-12):
            print(f"t = {t_val:.3f}, relative L2 error = {rel_l2:.3e}")

    rel_errors = np.asarray(rel_errors)
    rel_l2_all = np.sqrt(sq_error_sum / sq_ref_sum)

    print("\n===============================")
    print(f"Average relative L2 error: {np.mean(rel_errors):.3e}")
    print(f"Max relative L2 error:     {np.max(rel_errors):.3e}")
    print(f"Relative L2 on full x-y-t grid: {rel_l2_all:.3e}")
    print("===============================")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


# 👇生成高精度参考解
# 用周期 Fourier pseudospectral + 半隐式 Euler 生成传统数值参考解，
# 默认 256x256, dt=1e-3, t in [0,10]，保存为 reference_2d.npz。
# 解释：周期 Fourier pseudospectral方法来离散空间，时间推进使用的是 半隐式 Euler 格式。


# 👇误差计算：
# 所有保存时刻上的相对 L2 误差的平均值。rel_l2(t_j) = ||u_PINN(x,y,t_j) - u_ref(x,y,t_j)||_2/ ||u_ref(x,y,t_j)||_2
# Average relative L2 error:
# 所有保存时刻中最大的单时刻相对 L2 误差。  Max = max_j rel_l2(t_j)
# Max relative L2 error: 
# 这是 整个三维时空网格上的整体相对 L2 误差。shape: (101, 256, 256)
# Relative L2 on full x-y-t grid:

# 使用周期 Fourier pseudospectral方法来离散空间，时间推进使用半隐式Euler格式，生成高精度参考解。结果表明，所有保存时刻上的相对L2误差的平均值为5.01e-03；所有保存时刻中最大的单时刻相对L2误差为2.37e-02；整个三维时空网格上的整体相对L2误差为6.70e-03。