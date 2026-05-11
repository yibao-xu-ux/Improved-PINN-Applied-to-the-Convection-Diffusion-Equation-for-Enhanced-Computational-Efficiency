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
    velocity_x: float = 1.0
    velocity_y: float = 1.0
    diffusion: float = 1.0e-4
    length: float = 2.0 * np.pi
    wave_number: float = 3.0


@dataclass(frozen=True)
class TrainConfig:
    dt: float = 0.1
    t_end: float = 1.0

    n_layers: int = 5
    n_neurons: int = 64

    n_f: int = 5000
    n_b: int = 700
    n_i: int = 2500
    n_adap: int = 3000    # top-k中的k参数

    adam_iters: int = 4500
    adap_every: int = 900   # adam优化器每隔多少步加一次配点
    lbfgs_iters: int = 4500
    lr: float = 1.0e-3

    pde_weight: float = 1.0
    periodic_value_weight: float = 10.0
    periodic_grad_weight: float = 1.0
    ic_weight: float = 10.0

    print_every: int = 100
    save_dir: str = os.path.join(BASE_DIR, "models_harmonic_wave_2d")

    @property
    def n_steps(self) -> int:
        return int(round(self.t_end / self.dt))


PROBLEM = ProblemConfig()
TRAIN = TrainConfig()
os.makedirs(TRAIN.save_dir, exist_ok=True)


def exact_solution(x, y, t):
    k = PROBLEM.wave_number
    phase_x = k * (x - PROBLEM.velocity_x * t)
    phase_y = k * (y - PROBLEM.velocity_y * t)
    decay = torch.exp(-2.0 * PROBLEM.diffusion * k**2 * t)
    return 0.5 + 0.5 * torch.sin(phase_x) * torch.sin(phase_y) * decay


def initial_condition(x, y):
    return exact_solution(x, y, torch.zeros_like(x))


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(3, TRAIN.n_neurons), nn.Tanh()]
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

    def forward(self, xyt):
        return self.net(xyt)


class SingleStepPINN:
    def __init__(self, t0, xy_ic, phi_ic, prev_model=None, step_id=0):
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
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-11,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        self.xy_ic = xy_ic
        self.phi_ic = phi_ic
        self.X_ic = self._make_xyt(
            self.xy_ic[:, 0:1],
            self.xy_ic[:, 1:2],
            torch.full_like(self.xy_ic[:, 0:1], self.t0),
        )

        self.X_f = self.sample_f()
        self.X_x0, self.X_xL, self.X_y0, self.X_yL = self.sample_periodic_boundary()

    @staticmethod
    def _make_xyt(x, y, t):
        return torch.cat([x, y, t], dim=1).to(DEVICE)

    def sample_f(self):
        points = lhs(3, TRAIN.n_f)
        points[:, 0] = points[:, 0] * PROBLEM.length
        points[:, 1] = points[:, 1] * PROBLEM.length
        points[:, 2] = points[:, 2] * TRAIN.dt + self.t0
        return torch.tensor(points, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sample_periodic_boundary(self):
        s = torch.rand(TRAIN.n_b, 1, dtype=DTYPE, device=DEVICE) * PROBLEM.length
        t = torch.rand(TRAIN.n_b, 1, dtype=DTYPE, device=DEVICE) * TRAIN.dt + self.t0
        zeros = torch.zeros_like(s)
        length = torch.full_like(s, PROBLEM.length)

        X_x0 = self._make_xyt(zeros, s, t).detach().requires_grad_(True)
        X_xL = self._make_xyt(length, s, t).detach().requires_grad_(True)
        X_y0 = self._make_xyt(s, zeros, t).detach().requires_grad_(True)
        X_yL = self._make_xyt(s, length, t).detach().requires_grad_(True)
        return X_x0, X_xL, X_y0, X_yL

    def pde_residual(self, X):
        phi = self.model(X)
        grads = torch.autograd.grad(
            phi,
            X,
            torch.ones_like(phi),
            create_graph=True,
        )[0]
        phi_x = grads[:, 0:1]
        phi_y = grads[:, 1:2]
        phi_t = grads[:, 2:3]

        phi_xx = torch.autograd.grad(
            phi_x,
            X,
            torch.ones_like(phi_x),
            create_graph=True,
        )[0][:, 0:1]
        phi_yy = torch.autograd.grad(
            phi_y,
            X,
            torch.ones_like(phi_y),
            create_graph=True,
        )[0][:, 1:2]

        return (
            phi_t
            + PROBLEM.velocity_x * phi_x
            + PROBLEM.velocity_y * phi_y
            - PROBLEM.diffusion * (phi_xx + phi_yy)
        )

    def _value_and_grad(self, X):
        phi = self.model(X)
        grad = torch.autograd.grad(
            phi,
            X,
            torch.ones_like(phi),
            create_graph=True,
        )[0]
        return phi, grad

    def periodic_loss(self):
        phi_x0, grad_x0 = self._value_and_grad(self.X_x0)
        phi_xL, grad_xL = self._value_and_grad(self.X_xL)
        phi_y0, grad_y0 = self._value_and_grad(self.X_y0)
        phi_yL, grad_yL = self._value_and_grad(self.X_yL)

        loss_value = torch.mean((phi_x0 - phi_xL) ** 2) + torch.mean(
            (phi_y0 - phi_yL) ** 2
        )
        loss_grad = torch.mean((grad_x0[:, 0:1] - grad_xL[:, 0:1]) ** 2)
        loss_grad = loss_grad + torch.mean((grad_y0[:, 1:2] - grad_yL[:, 1:2]) ** 2)
        return loss_value, loss_grad

    def adaptive_refine(self):
        X_cand = self.sample_f()
        residual = self.pde_residual(X_cand).abs().squeeze()
        _, idx = torch.topk(residual, TRAIN.n_adap)
        X_new = X_cand[idx].detach().requires_grad_(True)
        self.X_f = torch.cat([self.X_f, X_new], dim=0)

    def loss(self):
        loss_pde = torch.mean(self.pde_residual(self.X_f) ** 2)
        loss_periodic_value, loss_periodic_grad = self.periodic_loss()
        loss_ic = torch.mean((self.model(self.X_ic) - self.phi_ic) ** 2)

        return (
            TRAIN.pde_weight * loss_pde
            + TRAIN.periodic_value_weight * loss_periodic_value
            + TRAIN.periodic_grad_weight * loss_periodic_grad
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


def sample_initial_points():
    points = lhs(2, TRAIN.n_i)
    points[:, 0] = points[:, 0] * PROBLEM.length
    points[:, 1] = points[:, 1] * PROBLEM.length
    return torch.tensor(points, dtype=DTYPE, device=DEVICE)


def solve():
    xy_ic = sample_initial_points()
    phi_ic = initial_condition(xy_ic[:, 0:1], xy_ic[:, 1:2])

    prev_model = None
    total_start = time.time()

    for n in range(TRAIN.n_steps):
        t0 = n * TRAIN.dt
        t1 = t0 + TRAIN.dt
        print(f"\n=== Step {n + 1}/{TRAIN.n_steps}, t in [{t0:.3f}, {t1:.3f}] ===")

        pinn = SingleStepPINN(
            t0=t0,
            xy_ic=xy_ic,
            phi_ic=phi_ic,
            prev_model=prev_model,
            step_id=n + 1,
        )
        prev_model = pinn.train()

        t_next = torch.full_like(xy_ic[:, 0:1], t1)
        X_next = torch.cat([xy_ic[:, 0:1], xy_ic[:, 1:2], t_next], dim=1).to(DEVICE)
        with torch.no_grad():
            phi_ic = prev_model(X_next).detach()

    total_elapsed = time.time() - total_start
    print("\n==============================")
    print(f"Total training time = {total_elapsed:.2f} s")
    print("==============================")


if __name__ == "__main__":
    solve()


# 👉第一个时间步下降地好缓慢，似乎是1d、2d所有例子中最缓慢的一个例子。
# 第一步最终只下降到5.774e-04数量级
