# large的时候效果不太行，该文件作废
import torch
import torch.nn as nn
import numpy as np
import os
from pyDOE import lhs

# =====================================================
# 全局配置
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

a = 1.0
kappa = 0.01 / np.pi

DT = 0.1
T_END = 1.0  # 2.0
N_STEPS = int(T_END / DT)

N_LAYERS = 4
N_NEURONS = 40

N_F = 3000
N_B = 400
N_I = 200
N_ADAP = 1000

ADAM_ITERS = 4000
ADAP_EVERY = 800
LBFGS_ITERS = 5000
LR = 1e-3

PRINT_EVERY = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "models_stm")
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# 网络（Xavier 初始化）
# =====================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, N_NEURONS), nn.Tanh()]
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
            self.model.load_state_dict(prev_model.state_dict())  # 参数继承

        self.adam = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.adam,
            gamma=0.9995  # ⭐关键参数
        )
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=LBFGS_ITERS,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        self.X_ic = X_ic
        self.u_ic = u_ic

        self.X_f = self.sample_f()
        self.X_l, self.X_r = self.sample_bc()

    # ---------- sampling ----------
    def sample_f(self):
        X = lhs(2, N_F)
        X[:, 0] = X[:, 0] * 2 - 1
        X[:, 1] = X[:, 1] * DT + self.t0
        return torch.tensor(X, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sample_bc(self):
        t = torch.rand(N_B, 1) * DT + self.t0
        X_l = torch.cat([-torch.ones_like(t), t], 1).to(DEVICE)  # -1
        X_r = torch.cat([ torch.ones_like(t), t], 1).to(DEVICE)  # 1
        return X_l, X_r

    # ---------- physics ----------
    def pde_residual(self, X):
        u = self.model(X)
        grads = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = torch.autograd.grad(
            u_x, X, torch.ones_like(u_x), create_graph=True
        )[0][:, 0:1]
        return u_t + a * u_x - kappa * u_xx

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
        loss_bc = (
            torch.mean(self.model(self.X_l) ** 2)
            + torch.mean(self.model(self.X_r) ** 2)   # x=-1/1时,u=0
        )
        loss_ic = torch.mean((self.model(self.X_ic) - self.u_ic) ** 2)
        return loss_pde + 10 * loss_bc + 10 * loss_ic
    
    # -----------tool func----------------
    def _log_and_step(self, loss):
        self.iter += 1
        if self.iter % PRINT_EVERY == 0:
            print(f"Iter {self.iter}, loss = {loss.item():.3e}")

    # ---------- training ----------
    def train(self):
        # Adam + adaptive
        for it in range(ADAM_ITERS):
            self.adam.zero_grad()
            loss = self.loss()
            loss.backward()
            self.adam.step()
            self.scheduler.step()   # 👈 学习率指数衰减

            if it % ADAP_EVERY == 0 and it>0: # it < 0.5 * ADAM_ITERS:  
                self.adaptive_refine()   

            self._log_and_step(loss)

        # LBFGS 精修（固定点集）
        def closure():
            self.lbfgs.zero_grad()
            loss = self.loss()
            loss.backward()
            self._log_and_step(loss)  # lbfgs没有iter的概念，将调用closure的次数作为迭代数
            return loss

        self.lbfgs.step(closure)

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
    x = torch.linspace(-1, 1, N_I).view(-1, 1)
    t = torch.zeros_like(x)
    X_ic = torch.cat([x, t], 1).to(DEVICE)
    u_ic = -torch.sin(np.pi * x).to(DEVICE)

    prev_model = None

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

        t_next = torch.full_like(x, t0 + DT)
        X_ic = torch.cat([x, t_next], 1).to(DEVICE)
        with torch.no_grad():
            u_ic = prev_model(X_ic)

# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":
    solve()
