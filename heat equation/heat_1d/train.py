import torch
import torch.nn as nn
from collections import OrderedDict
import math
import numpy as np
import time

import os
# 当前脚本所在目录（不是运行目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 模型保存路径
MODEL_PATH = os.path.join(BASE_DIR, "heat_1d.pth")

import torch
import numpy as np
import random

# # =============================
# # 设置随机种子
# # =============================
# SEED = 1234
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False # 关闭cudnn自动优化
# 👉后面的实验先不整随机种子了，感觉没必要

class Network(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, act=nn.Tanh):
        super().__init__()

        layers = []
        layers.append(("input", nn.Linear(in_dim, hidden)))
        layers.append(("act0", act()))

        for i in range(depth):
            layers.append((f"linear{i}", nn.Linear(hidden, hidden)))
            layers.append((f"act{i}", act()))

        layers.append(("output", nn.Linear(hidden, out_dim)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.net(x)


# =====================================================
# PINN
# =====================================================
class PINN:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Network(
            in_dim=2,
            hidden=20,
            out_dim=1,
            depth=6,
            act=nn.Tanh
        ).to(self.device)

        self.h = 0.05
        self.k = 0.05

        # ---------- grid ----------
        x = torch.arange(0, 1 + self.h, self.h)
        t = torch.arange(0, 40 + self.k, self.k)

        X, T = torch.meshgrid(x, t, indexing='ij')
        self.X_inside = torch.stack([X.flatten(), T.flatten()], dim=1)

        # ---------- boundary ----------
        bc1 = torch.stack(torch.meshgrid(x[:1], t, indexing='ij'), dim=-1).reshape(-1, 2)
        bc2 = torch.stack(torch.meshgrid(x[-1:], t, indexing='ij'), dim=-1).reshape(-1, 2)
        ic = torch.stack(torch.meshgrid(x, t[:1], indexing='ij'), dim=-1).reshape(-1, 2)

        self.X_boundary = torch.cat([bc1, bc2, ic], dim=0)

        u_bc1 = torch.zeros(len(bc1))
        u_bc2 = torch.zeros(len(bc2))
        u_ic = torch.sin(math.pi * ic[:, 0])

        self.U_boundary = torch.cat([u_bc1, u_bc2, u_ic]).unsqueeze(1)

        # ---------- device ----------
        self.X_inside = self.X_inside.to(self.device).requires_grad_(True)
        self.X_boundary = self.X_boundary.to(self.device)
        self.U_boundary = self.U_boundary.to(self.device)

        self.mse = nn.MSELoss()
        self.iter = 1

        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=20000,
            max_eval=20000,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )

    def loss_func(self):

        self.lbfgs.zero_grad()

        # ----- boundary loss -----
        pred_bc = self.model(self.X_boundary)
        loss_bc = self.mse(pred_bc, self.U_boundary)

        # ----- PDE loss -----
        u = self.model(self.X_inside)

        grad_u = torch.autograd.grad(
            u, self.X_inside,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = grad_u[:, 0]
        u_t = grad_u[:, 1]

        u_xx = torch.autograd.grad(
            u_x, self.X_inside,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        loss_eq = self.mse(u_t, 0.01 * u_xx)

        loss = loss_eq + loss_bc
        loss.backward()

        if self.iter % 100 == 0:
            print(f"{self.iter:5d}  loss={loss.item():.3e}")

        self.iter += 1
        return loss

    def train(self):
        print("采用 L-BFGS 优化器")
        self.model.train()
        self.lbfgs.step(self.loss_func)


# =====================================================
# main
# =====================================================
if __name__ == "__main__":  # 一定要写，否则运行test时import会运行train
    pinn = PINN()
    start=time.time()
    pinn.train()
    end=time.time()
    total_time = end-start
    print(total_time)
    torch.save(pinn.model.state_dict(), MODEL_PATH)
