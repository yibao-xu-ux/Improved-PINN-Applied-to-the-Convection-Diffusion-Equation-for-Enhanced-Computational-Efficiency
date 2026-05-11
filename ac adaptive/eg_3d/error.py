import os

import numpy as np
import torch

from plot import REFERENCE_PATH, evaluate_pinn, get_step_id, load_model_3d


def main():
    if not os.path.exists(REFERENCE_PATH):
        raise FileNotFoundError(
            f"{REFERENCE_PATH} not found. Run generate_reference.py first."
        )

    data = np.load(REFERENCE_PATH)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    tt = data["tt"]
    uu = data["uu"]
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    rel_errors = []
    sq_error_sum = 0.0
    sq_ref_sum = 0.0
    model_cache = {}

    for j, t_val in enumerate(tt):
        t_val = float(t_val)
        step_id = get_step_id(t_val)
        if step_id not in model_cache:
            model_cache[step_id] = load_model_3d(step_id)

        u_pred = evaluate_pinn(model_cache[step_id], x_grid, y_grid, z_grid, t_val)
        u_ref = uu[j]
        diff = u_pred - u_ref

        rel_l2 = np.linalg.norm(diff) / np.linalg.norm(u_ref)
        rel_errors.append(rel_l2)
        sq_error_sum += float(np.sum(diff**2))
        sq_ref_sum += float(np.sum(u_ref**2))

        if j == 0 or j == len(tt) - 1 or np.isclose(t_val * 10, round(t_val * 10), atol=1e-10):
            print(f"t = {t_val:.3f}, relative L2 error = {rel_l2:.3e}")

    rel_errors = np.asarray(rel_errors)
    rel_l2_all = np.sqrt(sq_error_sum / sq_ref_sum)

    print("\n===============================")
    print(f"Average relative L2 error: {np.mean(rel_errors):.3e}")
    print(f"Max relative L2 error:     {np.max(rel_errors):.3e}")
    print(f"Relative L2 on full x-y-z-t grid: {rel_l2_all:.3e}")
    print("===============================")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()



# 仍然使用周期Fourier pseudospectral+半隐式Euler生成高精度参考解。结果表明，所有保存时刻上的相对L2误差的平均值为9.31e-04；所有保存时刻中最大的单时刻相对L2误差为3.10e-03；整个四维时空网格上的整体相对L2误差为1.02e-03。