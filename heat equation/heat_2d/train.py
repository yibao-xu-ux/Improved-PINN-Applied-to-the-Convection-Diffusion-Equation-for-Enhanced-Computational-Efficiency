import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import math
import os
import time

# ---------- save path ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heat_2d.pth")


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


class PINN:
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Network(
            in_dim=3,
            hidden=32,
            out_dim=1,
            depth=6,
            act=nn.Tanh
        ).to(self.device)

        # ---------- parameters ----------
        self.hx = 0.05
        self.hy = 0.05
        self.dt = 0.05
        self.alpha = 0.01

        # ---------- grid ----------
        x = torch.arange(0, 1 + self.hx, self.hx)
        y = torch.arange(0, 2 + self.hy, self.hy)
        t = torch.arange(0, 3 + self.dt, self.dt)

        X, Y, T = torch.meshgrid(x, y, t, indexing='ij')
        self.X_inside = torch.stack([X.flatten(), Y.flatten(), T.flatten()], dim=1)

        # ---------- boundary + initial ----------
        def plane(a, b, c):
            A, B = torch.meshgrid(a, b, indexing='ij')
            return torch.stack([c(A, B), A.flatten(), B.flatten()], dim=1)

        bc_x0 = torch.stack(torch.meshgrid(x[:1], y, t, indexing='ij'), dim=-1).reshape(-1, 3)
        bc_x1 = torch.stack(torch.meshgrid(x[-1:], y, t, indexing='ij'), dim=-1).reshape(-1, 3)
        bc_y0 = torch.stack(torch.meshgrid(x, y[:1], t, indexing='ij'), dim=-1).reshape(-1, 3)
        bc_y2 = torch.stack(torch.meshgrid(x, y[-1:], t, indexing='ij'), dim=-1).reshape(-1, 3)
        ic = torch.stack(torch.meshgrid(x, y, t[:1], indexing='ij'), dim=-1).reshape(-1, 3)

        self.X_boundary = torch.cat([bc_x0, bc_x1, bc_y0, bc_y2, ic], dim=0)

        u_bc = torch.zeros(len(bc_x0) + len(bc_x1) + len(bc_y0) + len(bc_y2))
        u_ic = torch.sin(math.pi * ic[:, 0]) * torch.sin(math.pi * ic[:, 1] / 2)

        self.U_boundary = torch.cat([u_bc, u_ic]).unsqueeze(1)

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

    # =================================================
    # loss
    # =================================================
    def loss_func(self):

        self.lbfgs.zero_grad()

        # ----- boundary -----
        pred_bc = self.model(self.X_boundary)
        loss_bc = self.mse(pred_bc, self.U_boundary)

        # ----- PDE -----
        u = self.model(self.X_inside)

        grad_u = torch.autograd.grad(
            u, self.X_inside,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        u_x = grad_u[:, 0]
        u_y = grad_u[:, 1]
        u_t = grad_u[:, 2]

        u_xx = torch.autograd.grad(
            u_x, self.X_inside,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        u_yy = torch.autograd.grad(
            u_y, self.X_inside,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]

        residual = u_t - self.alpha * (u_xx + u_yy)

        # t_weight = torch.exp(0.05 * self.X_inside[:, 2])   # 时间权重⏱️
        # loss_eq = (t_weight * residual ** 2).mean() # 内部损失加上了时间权重

        loss_eq = (residual ** 2).mean()

        loss = loss_eq + loss_bc
        loss.backward()

        if self.iter % 100 == 0:
            print(f"{self.iter:5d}  loss={loss.item():.3e}")

        self.iter += 1
        return loss

    # =================================================
    # train
    # =================================================
    def train(self):          # 只使用lbfgs优化器
        print("采用 L-BFGS 优化器")
        self.model.train()
        self.lbfgs.step(self.loss_func)


# =====================================================
# main
# =====================================================
if __name__ == "__main__":

    pinn = PINN()

    start = time.time()
    pinn.train()
    end = time.time()

    print("total time:", end - start)

    torch.save(pinn.model.state_dict(), MODEL_PATH)

