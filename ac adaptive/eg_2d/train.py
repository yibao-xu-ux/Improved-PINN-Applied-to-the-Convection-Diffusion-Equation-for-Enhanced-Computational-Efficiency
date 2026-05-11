import torch
import torch.nn as nn
import time
import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

DT = 1.0
N_STEPS = 10
N_LAYERS = 6
N_NEURONS = 100

N_COLLOCATIONS = 20000
N_INITIAL = 1000
N_BOUND = 1000
IC_WEIGHT = 100.0  # 初值权重

BC_DERIV_WEIGHT = 1.0  # enforce periodic first derivatives

ADAM_ITER = 1000
PRINT_ITER = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
MODEL_DIR = os.path.join(BASE_DIR, "2D_10_miniModel")

# -----ac参数-------
LAMBDA = 10.0
EPS = 0.025

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, depth):
        super().__init__()
        layers = [
            nn.Linear(input_size, hidden_size, dtype=DTYPE),
            nn.Tanh()
        ]
        for _ in range(depth):
            layers.append(nn.Linear(hidden_size, hidden_size, dtype=DTYPE))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size, dtype=DTYPE))
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
# 单时间步 PINN（2D Allen–Cahn）
# =====================================================
class SingleStepPINN2D:
    def __init__(self, x_f, x_init, x_bx1, x_bx2, x_by1, x_by2, step_id):
        self.step_id = step_id
        self.model = Network(3, N_NEURONS, 1, N_LAYERS).to(DEVICE)

        self.x_f = x_f
        self.x_init = x_init
        self.x_bx1 = x_bx1
        self.x_bx2 = x_bx2
        self.x_by1 = x_by1
        self.x_by2 = x_by2

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

        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]

        u_xx = torch.autograd.grad(
            u_x, self.x_f,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True
        )[0][:, 0:1]

        u_yy = torch.autograd.grad(
            u_y, self.x_f,
            grad_outputs=torch.ones_like(u_y),
            create_graph=True
        )[0][:, 1:2]

        laplace_u = u_xx + u_yy

        f = u_t - LAMBDA * (EPS**2 * laplace_u - u**3 + u)    # √
        return torch.mean(f**2)

    # -----------------------------
    # 周期边界
    # -----------------------------
    def loss_bc(self):
        loss_x = torch.mean((self.model(self.x_bx1) - self.model(self.x_bx2))**2)
        loss_y = torch.mean((self.model(self.x_by1) - self.model(self.x_by2))**2)
        return loss_x + loss_y

    def loss_bc_deriv(self):
        u_x1 = self.model(self.x_bx1)
        u_x2 = self.model(self.x_bx2)
        grad_x1 = torch.autograd.grad(
            u_x1,
            self.x_bx1,
            grad_outputs=torch.ones_like(u_x1),
            create_graph=True
        )[0]
        grad_x2 = torch.autograd.grad(
            u_x2,
            self.x_bx2,
            grad_outputs=torch.ones_like(u_x2),
            create_graph=True
        )[0]
        loss_dx = torch.mean((grad_x1[:, 0:1] - grad_x2[:, 0:1])**2)

        u_y1 = self.model(self.x_by1)
        u_y2 = self.model(self.x_by2)
        grad_y1 = torch.autograd.grad(
            u_y1,
            self.x_by1,
            grad_outputs=torch.ones_like(u_y1),
            create_graph=True
        )[0]
        grad_y2 = torch.autograd.grad(
            u_y2,
            self.x_by2,
            grad_outputs=torch.ones_like(u_y2),
            create_graph=True
        )[0]
        loss_dy = torch.mean((grad_y1[:, 1:2] - grad_y2[:, 1:2])**2)

        return loss_dx + loss_dy

    # -----------------------------
    # 初值
    # -----------------------------
    def loss_init(self):
        u_pred = self.model(self.x_init[:, :3])
        u_true = self.x_init[:, 3:4]
        return torch.mean((u_pred - u_true)**2)

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
    def save(self, folder=MODEL_DIR):
        os.makedirs(folder, exist_ok=True)
        path = f"{folder}/step_{self.step_id:03d}.pth"
        torch.save(self.model.state_dict(), path)

    # -----------------------------
    # 训练
    # -----------------------------
    def train(self):
        print(f"\n>>> Training step {self.step_id}")
        start_time = time.time()

        for _ in range(ADAM_ITER):
            self.adam.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.adam.step()
            self.iter += 1

            if self.iter % PRINT_ITER == 0:
                print(f"Iter {self.iter}, loss = {loss.item():.3e}")

        def closure():
            self.lbfgs.zero_grad()
            loss = self.total_loss()
            loss.backward()
            self.iter += 1
            self.last_lbfgs_loss = loss.item()
            if self.iter % PRINT_ITER == 0:
                print(f"[LBFGS]  iter {self.iter:6d} | loss = {loss.item():.3e}")
            return loss

        self.lbfgs.step(closure)
        self.save()

        elapsed = time.time() - start_time
        print(f"[L-BFGS final loss] {self.last_lbfgs_loss:.3e}")
        print(f"Step {self.step_id} time: {elapsed:.2f}s")

    def predict(self, x, y, t):
        xt = torch.cat([x, y, t], dim=1)
        with torch.no_grad():
            return self.model(xt)


# =====================================================
# 时间自适应管理器（Method II）
# =====================================================
class TimeAdaptivePINN2D:
    def __init__(self):
        self.models = []

    @staticmethod
    def initial_condition(x, y):
        r = torch.sqrt((x - 0.5)**2 + (y - 0.5)**2)    
        return torch.tanh((0.35 - r) / (2 * EPS))     # √

    def train(self):
        total_start = time.time()
        prev_model = None

        for k in range(N_STEPS):
            t0 = k * DT

            # collocation
            x = torch.rand(N_COLLOCATIONS, 1, device=DEVICE)
            y = torch.rand(N_COLLOCATIONS, 1, device=DEVICE)
            t = torch.rand(N_COLLOCATIONS, 1, device=DEVICE) * DT + t0
            x_f = torch.cat([x, y, t], 1).requires_grad_(True)

            # initial
            xi = torch.rand(N_INITIAL, 1, device=DEVICE)
            yi = torch.rand(N_INITIAL, 1, device=DEVICE)
            ti = torch.full_like(xi, t0)

            if k == 0:
                ui = self.initial_condition(xi, yi)
            else:
                ui = prev_model.predict(xi, yi, ti).detach()

            x_init = torch.cat([xi, yi, ti, ui], 1)

            # periodic boundary
            tb = torch.rand(N_BOUND, 1, device=DEVICE) * DT + t0
            xb = torch.rand(N_BOUND, 1, device=DEVICE)
            yb = torch.rand(N_BOUND, 1, device=DEVICE)

            x_bx1 = torch.cat([torch.zeros_like(tb), yb, tb], 1).requires_grad_(True)
            x_bx2 = torch.cat([torch.ones_like(tb), yb, tb], 1).requires_grad_(True)
            x_by1 = torch.cat([xb, torch.zeros_like(tb), tb], 1).requires_grad_(True)
            x_by2 = torch.cat([xb, torch.ones_like(tb), tb], 1).requires_grad_(True)

            model = SingleStepPINN2D(
                x_f, x_init,
                x_bx1, x_bx2,
                x_by1, x_by2,
                k + 1
            )

            if prev_model is not None:
                model.model.load_state_dict(prev_model.model.state_dict())

            model.train()
            self.models.append(model)
            prev_model = model

        print(f"\nTotal training time: {time.time() - total_start:.2f}s")


# =====================================================
if __name__ == "__main__":
    pinn = TimeAdaptivePINN2D()
    pinn.train()


'''
[L-BFGS final loss] 4.847e-03
Step 1 time: 130.17s
[L-BFGS final loss] 3.459e-04
Step 2 time: 155.92s
[L-BFGS final loss] 5.725e-04
Step 3 time: 38.16s
[L-BFGS final loss] 1.677e-04
Step 4 time: 132.12s
[L-BFGS final loss] 1.073e-04
Step 5 time: 72.27s
[L-BFGS final loss] 9.750e-05
Step 6 time: 98.93s
[L-BFGS final loss] 1.260e-04
Step 7 time: 87.73s
[L-BFGS final loss] 1.029e-04
Step 8 time: 160.80s
[L-BFGS final loss] 2.145e-04
Step 9 time: 147.47s
[L-BFGS final loss] 1.066e-02
Step 10 time: 141.95s
Total training time: 1167.52s
👉有一点慢,差不多20min了
'''

'''最新/codex改了之后:
[L-BFGS final loss] 6.303e-05
Step 1 time: 418.82s
[L-BFGS final loss] 8.479e-06
Step 2 time: 179.65s
[L-BFGS final loss] 4.323e-06
Step 3 time: 244.72s
[L-BFGS final loss] 4.581e-06
Step 4 time: 128.27s
[L-BFGS final loss] 5.135e-06
Step 5 time: 187.24s
[L-BFGS final loss] 4.485e-06
Step 6 time: 249.30s
[L-BFGS final loss] 7.873e-06
Step 7 time: 166.87s
[L-BFGS final loss] 2.256e-05
Step 8 time: 149.82s
[L-BFGS final loss] 2.397e-05
Step 9 time: 326.97s
[L-BFGS final loss] 5.426e-05
Step 10 time: 345.61s
Total training time: 2398.78s
代码优化后训练有点慢,正常，精度和配点数都增加了，周期性边界条件还增加了导数的部分。速度很慢但是精度很高。
'''

