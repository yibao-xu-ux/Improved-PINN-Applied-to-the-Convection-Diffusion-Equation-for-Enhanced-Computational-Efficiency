import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from adaptive_1d_small import PINN 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_stm_small")
DT = 0.1
T_END = 1.0
N_STEPS = int(T_END / DT)

NX = 100
NT = 100


def analytical_solution(x, t, a=1, kappa=0.1/np.pi):
    N = 800
    result = 0
    t1 = np.sinh(a / (2 * kappa))
    t2 = np.cosh(a / (2 * kappa))
    part1 = 0
    part2 = 0
    for p in range(N + 1):
        term_1 = ((-1)**p * (2 * p) * np.sin(np.pi * p * x) * np.exp(-kappa * p**2 * np.pi**2 * t)) / \
                 (a**4 + 8 * (a * kappa * np.pi)**2 * (p**2 + 1) + 16 * (np.pi * kappa)**4 * (p**2 - 1)**2)
        term_2 = ((-1)**p * (2 * p + 1) * np.cos((2 * p + 1) / 2 * np.pi * x) * np.exp(-kappa * (2 * p + 1)**2 / 4 * np.pi**2 * t)) / \
                 (a**4 + (a * kappa * np.pi)**2 * (8 * p**2 + 8 * p + 10) + (np.pi * kappa)**4 * (4 * p**2 + 4 * p - 3)**2)
        part1 += term_1
        part2 += term_2
    result = t1 * part1 + t2 * part2
    result *= 16 * np.pi**2 * kappa**3 * a * np.exp((a / (2 * kappa)) * (x - a * t / 2))
    return result

# =====================================================
# 加载模型
# =====================================================
_model_cache = {}

def load_model(step_id):
    if step_id not in _model_cache:
        model = PINN().to(DEVICE)
        path = os.path.join(MODEL_DIR, f"step_{step_id:03d}.pth")
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _model_cache[step_id] = model
    return _model_cache[step_id]

# =====================================================
# 计算全局相对L2误差
# =====================================================
def compute_relative_L2_error():
    # 生成网格
    x_vals = np.linspace(-1, 1, NX)
    t_vals = np.linspace(0, 1, NT)
    X, T = np.meshgrid(x_vals, t_vals)  # 生成时间-空间网格

    # 将网格转化为列向量
    X = torch.tensor(X, dtype=torch.float32).view(-1, 1)  # (Nx * Nt, 1)
    T = torch.tensor(T, dtype=torch.float32).view(-1, 1)  # (Nx * Nt, 1)

    X_T = torch.cat([X, T], dim=1)  # (Nx * Nt, 2)

    u_pred = torch.zeros_like(X_T[:, 0]).to(DEVICE)  # 确保u_pred与model在相同的设备上(GPU)

    t_np = T.squeeze().cpu().numpy()  # 这里会涉及到从GPU到CPU的转换
    
    # 遍历每个模型步骤
    for step_id in range(1, N_STEPS + 1):
        t_l = (step_id - 1) * DT
        t_r = step_id * DT
        
        # 获取该时间步对应的掩码
        mask = (t_np >= t_l) & (t_np < t_r)
        if not np.any(mask):
            continue

        # 加载模型
        model = load_model(step_id)
        X_sub = X_T[mask].to(DEVICE)  # 确保X_sub在正确的设备上

        with torch.no_grad():
            # 保证u_pred和模型输出都在GPU上
            u_pred[mask] = model(X_sub).squeeze()  # 不要移到CPU，这样可以避免设备不匹配

    # 计算参考解并确保在相同设备
    u_ref = np.array([analytical_solution(x, t) for x, t in zip(X.squeeze().numpy(), T.squeeze().numpy())])
    u_ref_tensor = torch.tensor(u_ref, dtype=torch.float32).to(DEVICE)  # 确保u_ref_tensor在正确设备

    # 计算相对L2误差
    err = np.linalg.norm(u_pred.cpu().numpy() - u_ref_tensor.cpu().numpy())  # 将u_pred和u_ref_tensor移到CPU
    ref = np.linalg.norm(u_ref_tensor.cpu().numpy())  # 将u_ref_tensor移到CPU
    rel_l2 = err / ref

    return rel_l2, u_pred, u_ref


# =====================================================
# 绘制热力图
# =====================================================
# def plot_error_heatmap(u_pred, u_ref, Nx=100, Nt=100):
#     # 将u_pred和u_ref移到CPU，并转换为numpy数组
#     u_pred_cpu = u_pred.cpu().numpy() if u_pred.is_cuda else u_pred.numpy()
#     u_ref_cpu = u_ref  # u_ref_cpu保持在CPU上的numpy数组

#     # 计算绝对误差
#     error = np.abs(u_pred_cpu - u_ref_cpu)

#     # 绘制热力图
#     plt.imshow(error.reshape(Nt, Nx), extent=[-1, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
#     plt.colorbar(label="Absolute Error")
#     plt.title("Absolute Error Heatmap")
#     plt.xlabel("x")
#     plt.ylabel("t")
#     plt.savefig("1d_abs_error.png", dpi=200)
#     print("图片已保存")

# =====================================================
# 绘制相对误差热力图
# =====================================================
def plot_relative_error_heatmap(u_pred, u_ref, Nx=100, Nt=100):
    # 将u_pred和u_ref移到CPU，并转换为numpy数组
    u_pred_cpu = u_pred.cpu().numpy() if u_pred.is_cuda else u_pred.numpy()
    u_ref_cpu = u_ref  # u_ref_cpu保持在CPU上的numpy数组

    # 计算相对误差
    relative_error = np.abs(u_pred_cpu - u_ref_cpu) / np.abs(u_ref_cpu)

    # 绘制相对误差热力图
    plt.imshow(relative_error.reshape(Nt, Nx), extent=[-1, 1, 0, 1], origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="Relative Error")
    plt.title("Relative Error Heatmap")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.savefig("1d_relative_error.png", dpi=200)
    print("图片已保存")

if __name__ == "__main__":
    rel_l2, u_pred, u_ref = compute_relative_L2_error()
    print(f"Global Relative L2 Error: {rel_l2:.4e}")

    # plot_error_heatmap(u_pred, u_ref)
    plot_relative_error_heatmap(u_pred, u_ref)
