# =====================================================
# error.py -- STM-PINN correct relative L2 error (with cache)
# =====================================================
import torch
import numpy as np
import os
from adaptive_2d import PINN   # ⚠️ 保证和训练用的 PINN 定义一致

# =====================================================
# config
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_stm_2d_pe200")
DT = 0.1
T_END = 1.0
N_STEPS = int(T_END / DT)

NX = 100
NY = 100
NT = 100

# ---------- PDE parameters (must match training) ----------
phi = np.deg2rad(22.5)
a_vec = np.array([np.cos(phi), np.sin(phi)])
kappa = 0.005

# =====================================================
# exact solution
# =====================================================
def exact_solution(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]

    ax = a_vec[0] * t
    ay = a_vec[1] * t

    return (1.0 / (4.0 * t + 1.0)) * torch.exp(
        - ((x - ax) ** 2 + (y - ay) ** 2) / (kappa * (4.0 * t + 1.0))
    )

# =====================================================
# STM model loader with cache
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
# main
# =====================================================
if __name__ == "__main__":

    # ---------- uniform grid ----------
    x = torch.linspace(0.0, 1.0, NX)
    y = torch.linspace(0.0, 1.0, NY)
    t = torch.linspace(0.0, 1.0, NT)  # ✅ 修正为 [0,1]

    X, Y, T = torch.meshgrid(x, y, t, indexing="ij")

    X_all = torch.stack([
        X.reshape(-1),
        Y.reshape(-1),
        T.reshape(-1)
    ], dim=1).to(DEVICE)

    # ---------- exact solution ----------
    with torch.no_grad():
        u_exact = exact_solution(X_all)

    # ---------- STM prediction ----------
    u_pred = torch.zeros_like(u_exact)
    t_np = X_all[:, 2].cpu().numpy()

    for step_id in range(1, N_STEPS + 1):
        t_l = (step_id - 1) * DT
        t_r = step_id * DT

        mask = (t_np >= t_l) & (t_np < t_r)
        if not np.any(mask):
            continue

        model = load_model(step_id)
        X_sub = X_all[mask]

        with torch.no_grad():
            u_pred[mask] = model(X_sub)

    # ---------- relative L2 error (global) ----------
    err = torch.linalg.norm(u_pred - u_exact)
    ref = torch.linalg.norm(u_exact)
    rel_l2 = (err / ref).item()

    print("====================================")
    print(f"Global Relative L2 error (STM): {rel_l2:.6e}")

    # ---------- relative L2 error per STM interval ----------
    print("\nPer-time-slab Relative L2 errors:")
    for step_id in range(1, N_STEPS + 1):
        t_l = (step_id - 1) * DT
        t_r = step_id * DT
        mask = (t_np >= t_l) & (t_np < t_r)
        if not np.any(mask):
            continue

        err_k = torch.linalg.norm(u_pred[mask] - u_exact[mask])
        ref_k = torch.linalg.norm(u_exact[mask])
        rel_k = (err_k / ref_k).item()

        print(f"  step {step_id:02d}  t in [{t_l:.2f}, {t_r:.2f}]:  Rel L2 = {rel_k:.6e}")


'''
Global Relative L2 error (STM): 6.976402e-02
🟨Nx*Ny*Nt=100*100*100

Per-time-slab Relative L2 errors:
🟨Per-time-slab:Nx*Ny*Nt=100*100*100
  step 01  t in [0.00, 0.10]:  Rel L2 = 6.617769e-03
  step 02  t in [0.10, 0.20]:  Rel L2 = 7.627975e-03
  step 03  t in [0.20, 0.30]:  Rel L2 = 1.140217e-02
  step 04  t in [0.30, 0.40]:  Rel L2 = 1.407340e-02
  step 05  t in [0.40, 0.50]:  Rel L2 = 1.262161e-02
  step 06  t in [0.50, 0.60]:  Rel L2 = 1.347655e-02
  step 07  t in [0.60, 0.70]:  Rel L2 = 1.534812e-02
  step 08  t in [0.70, 0.80]:  Rel L2 = 1.700820e-02
  step 09  t in [0.80, 0.90]:  Rel L2 = 1.821889e-02
  step 10  t in [0.90, 1.00]:  Rel L2 = 1.874944e-02
  '''
