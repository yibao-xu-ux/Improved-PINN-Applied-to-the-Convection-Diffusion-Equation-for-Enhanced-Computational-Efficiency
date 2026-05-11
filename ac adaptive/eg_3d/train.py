import torch
import torch.nn as nn
import time
import os
from collections import OrderedDict


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# =====================================================
# 全局配置
# =====================================================
DT = 0.1
N_STEPS = 10
N_LAYERS = 6
N_NEURONS = 100

N_COLLOCATIONS = 20000
N_INITIAL = 1000
N_BOUND = 1000
IC_WEIGHT = 100.0
BC_DERIV_WEIGHT = 1.0  # also enforce periodic spatial derivatives

ADAM_ITER = 1000
PRINT_ITER = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
MODEL_DIR = os.path.join(BASE_DIR, "3D_10_miniModel")

LAMBDA = 10.0
EPS = 0.05


class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super().__init__()
        layers = [
            ("input", nn.Linear(input_size, hidden_size)),
            ("act_in", nn.Tanh())
        ]
        for i in range(depth):
            layers.append((f"hidden_{i}", nn.Linear(hidden_size, hidden_size)))
            layers.append((f"act_{i}", nn.Tanh()))
        layers.append(("output", nn.Linear(hidden_size, output_size)))
        self.net = nn.Sequential(OrderedDict(layers))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class SingleStepPINN3D:
    def __init__(self, x_f, x_init, b_pairs, step_id):
        self.step_id = step_id
        self.model = Network(4, N_NEURONS, 1, N_LAYERS).to(DEVICE)

        self.x_f = x_f
        self.x_init = x_init
        self.b_pairs = b_pairs

        self.iter = 0
        self.last_lbfgs_loss = None

        self.adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=5000,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe"
        )

    # -----------------------------
    # PDE residual
    # -----------------------------
    def loss_pde(self):
        u = self.model(self.x_f)
        grads = torch.autograd.grad(
            u, self.x_f,
            grad_outputs=torch.ones_like(u),
            create_graph=True
        )[0]
        u_x, u_y, u_z, u_t = grads[:, 0:1], grads[:, 1:2], grads[:, 2:3], grads[:, 3:4]
        u_xx = torch.autograd.grad(u_x, self.x_f, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, self.x_f, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
        u_zz = torch.autograd.grad(u_z, self.x_f, torch.ones_like(u_z), create_graph=True)[0][:, 2:3]
        lap = u_xx + u_yy + u_zz
        f = u_t - LAMBDA * (EPS**2 * lap - u**3 + u)    # √
        return torch.mean(f**2)

    # -----------------------------
    # 周期边界
    # -----------------------------
    def loss_bc(self):
        loss = 0.0
        for b1, b2 in self.b_pairs:
            loss += torch.mean((self.model(b1) - self.model(b2))**2)
        return loss

    def loss_bc_deriv(self):
        loss = 0.0
        for b1, b2 in self.b_pairs:
            u1 = self.model(b1)
            u2 = self.model(b2)
            grad1 = torch.autograd.grad(
                u1,
                b1,
                grad_outputs=torch.ones_like(u1),
                create_graph=True,
            )[0][:, :3]
            grad2 = torch.autograd.grad(
                u2,
                b2,
                grad_outputs=torch.ones_like(u2),
                create_graph=True,
            )[0][:, :3]
            loss += torch.mean((grad1 - grad2) ** 2)
        return loss

    # -----------------------------
    # 初值
    # -----------------------------
    def loss_init(self):
        pred = self.model(self.x_init[:, :4])
        true = self.x_init[:, 4:5]
        return torch.mean((pred - true)**2)

    def total_loss(self):
        return (
            self.loss_pde()
            + self.loss_bc()
            + BC_DERIV_WEIGHT * self.loss_bc_deriv()
            + IC_WEIGHT * self.loss_init()
        )

    # -----------------------------
    # 保存模型
    # -----------------------------
    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = f"{MODEL_DIR}/step_{self.step_id:03d}.pth"
        torch.save(self.model.state_dict(), path)

    # -----------------------------
    # 训练
    # -----------------------------
    def train(self):
        print(f"\n>>> Training step {self.step_id}")
        start = time.time()

        for _ in range(ADAM_ITER):
            self.adam.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.adam.step()
            self.iter += 1
            if self.iter % PRINT_ITER == 0:
                print(f"[Adam ] iter {self.iter:5d} | loss = {loss.item():.3e}")

        def closure():
            self.lbfgs.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.iter += 1
            self.last_lbfgs_loss = loss.item()
            if self.iter % PRINT_ITER == 0:
                print(f"[LBFGS] iter {self.iter:5d} | loss = {loss.item():.3e}")
            return loss

        self.lbfgs.step(closure)
        print(f"[L-BFGS final loss] {self.last_lbfgs_loss:.3e}")
        print(f"[L-BFGS final iter] {self.iter:.3e}")

        self.save()
        print(f"[Step {self.step_id}] time = {time.time() - start:.2f}s")

    def predict(self, x, y, z, t):
        with torch.no_grad():
            return self.model(torch.cat([x, y, z, t], 1))



class TimeAdaptivePINN3D:
    @staticmethod
    def initial_condition(x, y, z):
        r = torch.sqrt((x-0.5)**2 + (y-0.5)**2 + (z-0.5)**2)   # √
        return torch.tanh((0.35 - r) / (2 * EPS))

    def train(self):
        prev_model = None
        total_start = time.time()

        for k in range(N_STEPS):
            t0 = k * DT

            # collocation points
            xyz = torch.rand(N_COLLOCATIONS, 3, device=DEVICE)
            t = torch.rand(N_COLLOCATIONS, 1, device=DEVICE) * DT + t0
            x_f = torch.cat([xyz, t], 1).requires_grad_(True)

            # initial condition
            xi = torch.rand(N_INITIAL, 1, device=DEVICE)
            yi = torch.rand(N_INITIAL, 1, device=DEVICE)
            zi = torch.rand(N_INITIAL, 1, device=DEVICE)
            ti = torch.full_like(xi, t0)

            if k == 0:
                ui = self.initial_condition(xi, yi, zi)
            else:
                ui = prev_model.predict(xi, yi, zi, ti).detach()

            x_init = torch.cat([xi, yi, zi, ti, ui], 1)

            # periodic boundary
            tb = torch.rand(N_BOUND, 1, device=DEVICE) * DT + t0
            r2 = torch.rand(N_BOUND, 2, device=DEVICE)

            b_pairs = [
                (torch.cat([torch.zeros_like(tb), r2[:,0:1], r2[:,1:2], tb], 1),
                 torch.cat([torch.ones_like(tb),  r2[:,0:1], r2[:,1:2], tb], 1)),

                (torch.cat([r2[:,0:1], torch.zeros_like(tb), r2[:,1:2], tb], 1),
                 torch.cat([r2[:,0:1], torch.ones_like(tb),  r2[:,1:2], tb], 1)),

                (torch.cat([r2[:,0:1], r2[:,1:2], torch.zeros_like(tb), tb], 1),
                 torch.cat([r2[:,0:1], r2[:,1:2], torch.ones_like(tb),  tb], 1))
            ]
            b_pairs = [(b1.requires_grad_(True), b2.requires_grad_(True)) for b1, b2 in b_pairs]

            model = SingleStepPINN3D(x_f, x_init, b_pairs, k + 1)
            if prev_model is not None:
                model.model.load_state_dict(prev_model.model.state_dict())

            model.train()
            prev_model = model

        print(f"\nTotal training time = {time.time() - total_start:.2f}s")





if __name__ == "__main__":
    pinn = TimeAdaptivePINN3D()
    pinn.train()




'''
[L-BFGS final loss] 2.668e-04  👉最后/e-04数量级时优化得非常慢了,精修式
[L-BFGS final iter] 5.790e+03
[Step 1] time = 712.18s       
👉很慢,感觉可以适当宽松一点,等有数值解再说
--------------------------------
[L-BFGS final loss] 4.171e-05 👉精修式,下面应该都是
[L-BFGS final iter] 4.180e+03
[Step 2] time = 545.44s
-------------------------------
[L-BFGS final loss] 1.408e-05
[L-BFGS final iter] 3.137e+03
[Step 3] time = 440.22s
-------------------------------
[L-BFGS final loss] 1.464e-05
[L-BFGS final iter] 2.683e+03
[Step 4] time = 310.63s
-------------------------------
[L-BFGS final loss] 4.119e-05
[L-BFGS final iter] 7.760e+02
[Step 5] time = 84.46s
-------------------------------
[L-BFGS final loss] 6.665e-05
[L-BFGS final iter] 8.840e+02
[Step 6] time = 99.03s
-------------------------------
[L-BFGS final loss] 2.232e-05
[L-BFGS final iter] 3.876e+03
[Step 7] time = 469.76s
-------------------------------
[L-BFGS final loss] 3.193e-05
[L-BFGS final iter] 1.506e+03
[Step 8] time = 175.45s
-------------------------------
[L-BFGS final loss] 3.539e-05
[L-BFGS final iter] 2.804e+03
[Step 9] time = 342.59s
-------------------------------
[L-BFGS final loss] 1.560e-05
[L-BFGS final iter] 5.423e+03
[Step 10] time = 660.18s
-------------------------------
Total training time = 3845.66s
'''

'''
[L-BFGS final loss] 6.250e-04
[L-BFGS final iter] 6.260e+03
[Step 1] time = 409.34s
[L-BFGS final loss] 3.570e-05
[L-BFGS final iter] 6.391e+03
[Step 2] time = 415.65s
[L-BFGS final loss] 1.084e-05
[L-BFGS final iter] 6.373e+03
[Step 3] time = 417.04s
[L-BFGS final loss] 9.974e-06
[L-BFGS final iter] 4.350e+03
[Step 4] time = 277.10s
[L-BFGS final loss] 6.659e-06
[L-BFGS final iter] 5.496e+03
[Step 5] time = 354.68s
[L-BFGS final loss] 4.686e-05
[L-BFGS final iter] 1.013e+03
[Step 6] time = 47.15s
[L-BFGS final loss] 1.508e-04
[L-BFGS final iter] 1.007e+03
[Step 7] time = 47.14s
[L-BFGS final loss] 3.306e-04
[L-BFGS final iter] 1.010e+03
[Step 8] time = 46.81s
[L-BFGS final loss] 1.386e-05
[L-BFGS final iter] 4.609e+03
[Step 9] time = 293.68s
[L-BFGS final loss] 1.413e-05
[L-BFGS final iter] 2.539e+03
[Step 10] time = 169.30s

Total training time = 2479.21s
'''
