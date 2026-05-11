# =====================================================
# 2D STM-PINN for convection–diffusion (Pe=200)
# =====================================================
import torch
import torch.nn as nn
import numpy as np
import os
from pyDOE import lhs
import time

# =====================================================
# 全局配置
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ---------- PDE parameters ----------
phi = np.deg2rad(22.5)
a_vec = np.array([np.cos(phi), np.sin(phi)])   # convection velocity
kappa = 0.005                                  # Pe = 200

# ---------- time marching ----------
DT = 0.1
T_END = 1.0
N_STEPS = int(T_END / DT)

# ---------- network ----------
N_LAYERS = 4
N_NEURONS = 40

# ---------- sampling ----------
N_F = 3000
N_B = 400
N_I = 40       # per dimension → N_I^2 IC points
N_ADAP = 1000

# ---------- training ----------
ADAM_ITERS = 4000
ADAP_EVERY = 800
LBFGS_ITERS = 5000
LR = 1e-3
PRINT_EVERY = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "models_stm_2d_pe200")
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# PINN 网络
# =====================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(3, N_NEURONS), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(N_NEURONS, N_NEURONS), nn.Tanh()]
        layers.append(nn.Linear(N_NEURONS, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)
    
# ----------ground truth-----------
def exact_solution(X):
    """
    X: (N,3) -> (x,y,t)
    """
    x = X[:, 0:1]
    y = X[:, 1:2]
    t = X[:, 2:3]

    ax = a_vec[0] * t
    ay = a_vec[1] * t

    return (1.0 / (4.0 * t + 1.0)) * torch.exp(
        - ((x - ax)**2 + (y - ay)**2) / (kappa * (4.0 * t + 1.0))
    )

# =====================================================
# 单时间步 PINN
# =====================================================
class SingleStepPINN:
    def __init__(self, t0, X_ic, u_ic, prev_model=None, step_id=0):
        self.t0 = t0
        self.step_id = step_id
        self.iter = 0

        self.model = PINN().to(DEVICE)
        if prev_model is not None:
            self.model.load_state_dict(prev_model.state_dict())

        self.adam = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.adam, gamma=0.9995
        )
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=LBFGS_ITERS,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        self.X_ic = X_ic
        self.u_ic = u_ic

        self.X_f = self.sample_f()
        self.X_bc = self.sample_bc()
 


    # ---------- sampling ----------
    def sample_f(self):
        X = lhs(2, N_F)
        t = np.random.rand(N_F, 1) * DT + self.t0
        X = np.hstack([X, t])
        return torch.tensor(X, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sample_bc(self):
        t = torch.rand(N_B, 1) * DT + self.t0

        y = torch.rand_like(t)          # [0,1]
        x = torch.rand_like(t)          # [0,1]

        X_l = torch.cat([torch.zeros_like(t), y, t], 1)   # x = 0
        X_r = torch.cat([torch.ones_like(t),  y, t], 1)   # x = 1
        Y_b = torch.cat([x, torch.zeros_like(t), t], 1)   # y = 0
        Y_t = torch.cat([x, torch.ones_like(t),  t], 1)   # y = 1

        return torch.cat([X_l, X_r, Y_b, Y_t], 0).to(DEVICE)

    # ---------- PDE ----------
    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]

        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]

        u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        return (
            u_t
            + a_vec[0] * u_x
            + a_vec[1] * u_y
            - kappa * (u_xx + u_yy)
        )

    # ---------- adaptive ----------
    def adaptive_refine(self):
        X_cand = self.sample_f()
        res = self.pde_residual(X_cand).abs().squeeze()
        _, idx = torch.topk(res, N_ADAP)
        X_new = X_cand[idx].detach().requires_grad_(True)
        self.X_f = torch.cat([self.X_f, X_new], 0)

    # ---------- loss ----------
    def loss(self):
        loss_pde = torch.mean(self.pde_residual(self.X_f) ** 2)
        u_bc_pred = self.model(self.X_bc)
        u_bc_exact = exact_solution(self.X_bc)
        loss_bc = torch.mean((u_bc_pred - u_bc_exact) ** 2)
        loss_ic = torch.mean((self.model(self.X_ic) - self.u_ic) ** 2)
        return loss_pde + 10 * loss_bc + 10 * loss_ic

    def _log(self, loss):
        self.iter += 1
        if self.iter % PRINT_EVERY == 0:
            print(f"Iter {self.iter}, loss = {loss.item():.3e}")

    # ---------- training ----------
    def train(self):
        start_time = time.time()  # ⏱️ 单模型结束开始

        for it in range(ADAM_ITERS):
            self.adam.zero_grad()
            loss = self.loss()
            loss.backward()
            self.adam.step()
            self.scheduler.step()

            if it % ADAP_EVERY == 0 and it > 0:
                self.adaptive_refine()

            self._log(loss)

        def closure():
            self.lbfgs.zero_grad()
            loss = self.loss()
            loss.backward()
            self._log(loss)
            return loss

        self.lbfgs.step(closure)

        end_time = time.time()   # ⏱️ 单模型结束计时
        print(
            f"Step {self.step_id} training time = "
            f"{end_time - start_time:.2f} s"
        )
    
        self.save()
        return self.model

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(SAVE_DIR, f"step_{self.step_id:03d}.pth")
        )

# =====================================================
# 时间推进
# =====================================================
def solve():
    x = torch.linspace(0, 1, N_I)
    y = torch.linspace(0, 1, N_I)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    X_ic = torch.cat([
        X.reshape(-1,1),
        Y.reshape(-1,1),
        torch.zeros(X.numel(),1)
    ], 1).to(DEVICE)

    u_ic = exact_solution(X_ic)

    prev_model = None
    
    total_start_time = time.time()   # ⏱️ 总计时开始

    for n in range(N_STEPS):
        t0 = n * DT
        print(f"\n=== Step {n+1}/{N_STEPS}, t ∈ [{t0:.2f}, {t0+DT:.2f}] ===")

        pinn = SingleStepPINN(
            t0=t0,
            X_ic=X_ic,
            u_ic=u_ic,
            prev_model=prev_model,
            step_id=n+1
        )

        prev_model = pinn.train()

        t_next = torch.full((X.numel(),1), t0 + DT, device=DEVICE)
        X_ic = torch.cat([X_ic[:,0:2], t_next], 1)
        with torch.no_grad():
            u_ic = prev_model(X_ic)

    total_end_time = time.time()   # ⏱️  总计时结束
    print("\n==============================")
    print(
        f"Total training time = "
        f"{total_end_time - total_start_time:.2f} s"
    )
    print("==============================")

# =====================================================
if __name__ == "__main__":
    solve()
'''
lbfgs=-10和-12/wide=40:  # 原配置
step1,loss=1e-06,step=8800,time=297.22s

lbfgs=-8和-10/wide=40:
step1,loss=3e-06,step=7500,time=192.91s

lbfgs=-8和-10/wide=64:  
step1,loss=2e-06,step=9400,time=303.53s
# 👉加宽了效率反而不好(不过表达能力应该更强,如果由参考解,肉眼看和参考解不一样,考虑加宽)

lbfgs=-7和-9/wide=40:
step1,loss=3e-05(👉后面基本都这个数量级),step=5200,time=91.38s
------------------------------
Iter 4200, loss = 2.390e-05
Step 2 training time = 59.61 s
Iter 4000, loss = 3.305e-05
Step 3 training time = 48.67 s
Iter 4000, loss = 3.018e-05
Step 4 training time = 50.09 s
Iter 4300, loss = 1.223e-05
Step 5 training time = 71.50 s
Iter 4000, loss = 8.872e-06
Step 6 training time = 58.46 s
Iter 4000, loss = 1.015e-05
Step 7 training time = 61.24 s
Iter 4000, loss = 1.107e-05
Step 8 training time = 62.42 s
Iter 4000, loss = 1.135e-05
Step 9 training time = 60.23 s
Iter 4000, loss = 1.027e-05
Step 10 training time = 62.27 s
==============================
Total training time = 627.81 s
==============================
'''
