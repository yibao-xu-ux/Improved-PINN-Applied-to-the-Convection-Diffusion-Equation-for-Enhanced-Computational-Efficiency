import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


@dataclass(frozen=True)
class ProblemConfig:
    velocity: float = 1.0
    diffusion: float = 1.0e-4
    length: float = 2.0 * np.pi
    left_edge: float = 1.5
    right_edge: float = 3.5
    min_time_for_exact: float = 1.0e-12


@dataclass(frozen=True)
class TrainConfig:
    dt: float = 0.1
    t_end: float = 1.0

    n_layers: int = 4
    n_neurons: int = 50

    n_f: int = 1600    # 增加了小=每个时间区间的配点个数
    n_b: int = 300
    n_i: int = 500
    n_adap: int = 1600  # k

    adam_iters: int = 5000
    adap_every: int = 1000     # 
    lbfgs_iters: int = 6000
    lr: float = 1.0e-3

    pde_weight: float = 1.0
    bc_weight: float = 20.0   # 自动调大了权重
    ic_weight: float = 20.0

    print_every: int = 100
    save_dir: str = os.path.join(BASE_DIR, "models_square_wave")

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt))


PROBLEM = ProblemConfig()
TRAIN = TrainConfig()
os.makedirs(TRAIN.save_dir, exist_ok=True)


def exact_solution(x, t):
    t_safe = torch.clamp(t, min=PROBLEM.min_time_for_exact)
    shift = x - PROBLEM.velocity * t_safe
    scale = torch.sqrt(4.0 * PROBLEM.diffusion * t_safe)
    left = torch.erf((shift - PROBLEM.left_edge) / scale)
    right = torch.erf((shift - PROBLEM.right_edge) / scale)
    return 0.5 * (left - right)


def initial_condition(x):
    inside = (x > PROBLEM.left_edge) & (x < PROBLEM.right_edge)
    return torch.where(inside, torch.ones_like(x), torch.zeros_like(x))


def boundary_condition(x, t):
    return exact_solution(x, t)


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, TRAIN.n_neurons), nn.Tanh()]
        for _ in range(TRAIN.n_layers - 1):
            layers.extend([nn.Linear(TRAIN.n_neurons, TRAIN.n_neurons), nn.Tanh()])
        layers.append(nn.Linear(TRAIN.n_neurons, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


class SingleStepPINN:
    def __init__(self, t0, x_ic, phi_ic, prev_model=None, step_id=0):
        self.t0 = float(t0)
        self.step_id = step_id
        self.iter = 0

        self.model = PINN().to(DEVICE)
        if prev_model is not None:
            self.model.load_state_dict(prev_model.state_dict())

        self.adam = torch.optim.Adam(self.model.parameters(), lr=TRAIN.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.adam,
            gamma=0.9995,
        )
        self.lbfgs = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=TRAIN.lbfgs_iters,
            tolerance_grad=1.0e-10,
            tolerance_change=1.0e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        self.x_ic = x_ic
        self.phi_ic = phi_ic
        self.X_ic = self._make_xt(self.x_ic, torch.full_like(self.x_ic, self.t0))

        self.X_f = self.sample_f()
        self.X_left, self.phi_left, self.X_right, self.phi_right = self.sample_bc()

    @staticmethod
    def _make_xt(x, t):
        return torch.cat([x, t], dim=1).to(DEVICE)

    def sample_f(self):
        points = lhs(2, TRAIN.n_f)
        points[:, 0] = points[:, 0] * PROBLEM.length
        points[:, 1] = points[:, 1] * TRAIN.dt + self.t0
        return torch.tensor(points, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sample_bc(self):
        t = torch.rand(TRAIN.n_b, 1, dtype=DTYPE, device=DEVICE) * TRAIN.dt + self.t0
        x_left = torch.zeros_like(t)
        x_right = torch.full_like(t, PROBLEM.length)

        X_left = self._make_xt(x_left, t)
        X_right = self._make_xt(x_right, t)
        phi_left = boundary_condition(x_left, t)
        phi_right = boundary_condition(x_right, t)
        return X_left, phi_left, X_right, phi_right

    def pde_residual(self, X):
        phi = self.model(X)
        grads = torch.autograd.grad(
            phi,
            X,
            torch.ones_like(phi),
            create_graph=True,
        )[0]
        phi_x = grads[:, 0:1]
        phi_t = grads[:, 1:2]
        phi_xx = torch.autograd.grad(
            phi_x,
            X,
            torch.ones_like(phi_x),
            create_graph=True,
        )[0][:, 0:1]
        return phi_t + PROBLEM.velocity * phi_x - PROBLEM.diffusion * phi_xx

    def adaptive_refine(self):
        X_cand = self.sample_f()
        residual = self.pde_residual(X_cand).abs().squeeze()
        _, idx = torch.topk(residual, TRAIN.n_adap)
        X_new = X_cand[idx].detach().requires_grad_(True)
        self.X_f = torch.cat([self.X_f, X_new], dim=0)

    def loss(self):
        loss_pde = torch.mean(self.pde_residual(self.X_f) ** 2)
        loss_bc = (
            torch.mean((self.model(self.X_left) - self.phi_left) ** 2)
            + torch.mean((self.model(self.X_right) - self.phi_right) ** 2)
        )
        loss_ic = torch.mean((self.model(self.X_ic) - self.phi_ic) ** 2)

        return (
            TRAIN.pde_weight * loss_pde
            + TRAIN.bc_weight * loss_bc
            + TRAIN.ic_weight * loss_ic
        )

    def _log_and_step(self, loss):
        self.iter += 1
        if self.iter % TRAIN.print_every == 0:
            print(f"Iter {self.iter}, loss = {loss.item():.3e}")

    def train(self):
        start_time = time.time()

        for it in range(TRAIN.adam_iters):
            self.adam.zero_grad()
            loss = self.loss()
            loss.backward()
            self.adam.step()
            self.scheduler.step()

            if it > 0 and it % TRAIN.adap_every == 0:
                self.adaptive_refine()

            self._log_and_step(loss)

        def closure():
            self.lbfgs.zero_grad()
            loss = self.loss()
            loss.backward()
            self._log_and_step(loss)
            return loss

        self.lbfgs.step(closure)

        elapsed = time.time() - start_time
        print(f"Step {self.step_id} training time = {elapsed:.2f} s")

        self.save()
        return self.model

    def save(self):
        path = os.path.join(TRAIN.save_dir, f"step_{self.step_id:03d}.pth")
        torch.save(self.model.state_dict(), path)


def solve():
    x = torch.linspace(
        0.0,
        PROBLEM.length,
        TRAIN.n_i,
        dtype=DTYPE,
        device=DEVICE,
    ).view(-1, 1)
    phi_ic = initial_condition(x)

    prev_model = None
    total_start = time.time()

    for n in range(TRAIN.n_steps):
        t0 = n * TRAIN.dt
        t1 = t0 + TRAIN.dt
        print(f"\n=== Step {n + 1}/{TRAIN.n_steps}, t in [{t0:.3f}, {t1:.3f}] ===")

        pinn = SingleStepPINN(
            t0=t0,
            x_ic=x,
            phi_ic=phi_ic,
            prev_model=prev_model,
            step_id=n + 1,
        )
        prev_model = pinn.train()

        t_next = torch.full_like(x, t1)
        X_next = torch.cat([x, t_next], dim=1).to(DEVICE)
        with torch.no_grad():
            phi_ic = prev_model(X_next).detach()

    total_elapsed = time.time() - total_start
    print("\n==============================")
    print(f"Total training time = {total_elapsed:.2f} s")
    print("==============================")


if __name__ == "__main__":
    solve()
