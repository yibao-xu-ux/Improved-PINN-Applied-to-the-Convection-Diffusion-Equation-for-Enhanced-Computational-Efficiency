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
    left_edge: float = 1.0
    right_edge: float = 2.0
    initial_time: float = 0.0
    min_time_for_exact: float = 1.0e-12


@dataclass(frozen=True)
class TrainConfig:
    t_end: float = 1.0
    n_steps: int = 20

    n_layers: int = 5
    n_neurons: int = 64

    n_f: int = 6000
    n_b: int = 800
    n_i: int = 4000
    n_s: int = 6000
    n_adap: int = 4000     # k

    adam_iters: int = 6000
    adap_every: int = 1200   # m=6000/1200=5
    lbfgs_iters: int = 7000
    lr: float = 1.0e-3

    pde_weight: float = 1.0
    bc_weight: float = 20.0
    data_weight: float = 30.0
    range_weight: float = 5.0
    ic_weight: float = 100.0
    first_step_pde_weight: float = 2.0
    first_step_data_weight: float = 60.0
    first_step_range_weight: float = 10.0
    first_step_ic_weight: float = 200.0
    use_exact_step_initial: bool = True

    print_every: int = 100
    save_dir: str = os.path.join(BASE_DIR, "models_square_wave_2d_refined")

    @property
    def dt(self) -> float:
        return (self.t_end - PROBLEM.initial_time) / self.n_steps


PROBLEM = ProblemConfig()
TRAIN = TrainConfig()
os.makedirs(TRAIN.save_dir, exist_ok=True)


def exact_solution(x, y, t):
    t_safe = torch.clamp(t, min=PROBLEM.min_time_for_exact)
    scale = torch.sqrt(4.0 * PROBLEM.diffusion * t_safe)

    shifted_x = x - PROBLEM.velocity_x * t
    shifted_y = y - PROBLEM.velocity_y * t

    x_part = torch.erf((shifted_x - PROBLEM.left_edge) / scale) - torch.erf(
        (shifted_x - PROBLEM.right_edge) / scale
    )
    y_part = torch.erf((shifted_y - PROBLEM.left_edge) / scale) - torch.erf(
        (shifted_y - PROBLEM.right_edge) / scale
    )
    smooth_value = 0.25 * x_part * y_part
    return torch.where(t <= 0.0, discontinuous_initial_condition(x, y), smooth_value)


def discontinuous_initial_condition(x, y):
    inside_x = (x > PROBLEM.left_edge) & (x < PROBLEM.right_edge)
    inside_y = (y > PROBLEM.left_edge) & (y < PROBLEM.right_edge)
    inside = inside_x & inside_y
    return torch.where(inside, torch.ones_like(x), torch.zeros_like(x))


def initial_condition(x, y):
    return discontinuous_initial_condition(x, y)


def boundary_condition(x, y, t):
    return exact_solution(x, y, t)


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

    def set_step_start_time(self, t0):
        return None


class SingleStepPINN:
    def __init__(self, t0, xy_ic, phi_ic, prev_model=None, step_id=0):
        self.t0 = float(t0)
        self.step_id = step_id
        self.iter = 0
        self.last_loss_parts = None

        self.model = PINN().to(DEVICE)
        if prev_model is not None:
            self.model.load_state_dict(prev_model.state_dict())
        self.model.set_step_start_time(self.t0)

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

        self.xy_ic = xy_ic
        self.phi_ic = phi_ic
        self.X_ic = self._make_xyt(
            self.xy_ic[:, 0:1],
            self.xy_ic[:, 1:2],
            torch.full_like(self.xy_ic[:, 0:1], self.t0),
        )

        self.X_f = self.sample_f()
        self.X_b, self.phi_b = self.sample_bc()
        self.X_s, self.phi_s = self.sample_supervised_points()

    @staticmethod
    def _make_xyt(x, y, t):
        return torch.cat([x, y, t], dim=1).to(DEVICE)

    def sample_f(self):
        n_random = TRAIN.n_f // 2
        n_front = TRAIN.n_f - n_random

        random_points = lhs(3, n_random)
        random_points[:, 0] = random_points[:, 0] * PROBLEM.length
        random_points[:, 1] = random_points[:, 1] * PROBLEM.length
        random_points[:, 2] = random_points[:, 2] * TRAIN.dt + self.t0

        front_points = lhs(3, n_front)
        front_points[:, 2] = front_points[:, 2] * TRAIN.dt + self.t0
        front_selector = np.random.randint(0, 4, size=n_front)
        front_width = 8.0 * np.sqrt(PROBLEM.diffusion * max(self.t0 + TRAIN.dt, TRAIN.dt))
        front_width = max(front_width, 2.0e-2)

        moving_left_x = PROBLEM.left_edge + PROBLEM.velocity_x * front_points[:, 2]
        moving_right_x = PROBLEM.right_edge + PROBLEM.velocity_x * front_points[:, 2]
        moving_left_y = PROBLEM.left_edge + PROBLEM.velocity_y * front_points[:, 2]
        moving_right_y = PROBLEM.right_edge + PROBLEM.velocity_y * front_points[:, 2]

        front_points[:, 0] = front_points[:, 0] * PROBLEM.length
        front_points[:, 1] = front_points[:, 1] * PROBLEM.length
        jitter = (np.random.rand(n_front) - 0.5) * 2.0 * front_width

        x_left = front_selector == 0
        x_right = front_selector == 1
        y_left = front_selector == 2
        y_right = front_selector == 3
        front_points[x_left, 0] = moving_left_x[x_left] + jitter[x_left]
        front_points[x_right, 0] = moving_right_x[x_right] + jitter[x_right]
        front_points[y_left, 1] = moving_left_y[y_left] + jitter[y_left]
        front_points[y_right, 1] = moving_right_y[y_right] + jitter[y_right]
        front_points[:, 0] = np.clip(front_points[:, 0], 0.0, PROBLEM.length)
        front_points[:, 1] = np.clip(front_points[:, 1], 0.0, PROBLEM.length)

        points = np.vstack([random_points, front_points])
        return torch.tensor(points, dtype=DTYPE, device=DEVICE, requires_grad=True)

    def sample_bc(self):
        n_edge = TRAIN.n_b
        s = torch.rand(n_edge, 1, dtype=DTYPE, device=DEVICE) * PROBLEM.length
        t = torch.rand(n_edge, 1, dtype=DTYPE, device=DEVICE) * TRAIN.dt + self.t0

        zeros = torch.zeros_like(s)
        length = torch.full_like(s, PROBLEM.length)

        X_bottom = self._make_xyt(s, zeros, t)
        X_top = self._make_xyt(s, length, t)
        X_left = self._make_xyt(zeros, s, t)
        X_right = self._make_xyt(length, s, t)

        X_b = torch.cat([X_bottom, X_top, X_left, X_right], dim=0)
        phi_b = boundary_condition(
            X_b[:, 0:1],
            X_b[:, 1:2],
            X_b[:, 2:3],
        )
        return X_b, phi_b

    def sample_supervised_points(self):
        n_random = TRAIN.n_s // 4
        n_initial = TRAIN.n_s // 4
        n_front = TRAIN.n_s - n_random - n_initial

        random_points = lhs(3, n_random)
        random_points[:, 0] = random_points[:, 0] * PROBLEM.length
        random_points[:, 1] = random_points[:, 1] * PROBLEM.length
        random_points[:, 2] = random_points[:, 2] * TRAIN.dt + self.t0

        initial_points = lhs(3, n_initial)
        initial_points[:, 0] = initial_points[:, 0] * PROBLEM.length
        initial_points[:, 1] = initial_points[:, 1] * PROBLEM.length
        initial_points[:, 2] = self.t0 + initial_points[:, 2] * min(
            0.2 * TRAIN.dt,
            2.0e-3,
        )

        front_points = lhs(3, n_front)
        front_points[:, 2] = front_points[:, 2] * TRAIN.dt + self.t0
        selector = np.random.randint(0, 4, size=n_front)
        width = 12.0 * np.sqrt(PROBLEM.diffusion * max(self.t0 + TRAIN.dt, TRAIN.dt))
        width = max(width, 3.0e-2)

        left_x = PROBLEM.left_edge + PROBLEM.velocity_x * front_points[:, 2]
        right_x = PROBLEM.right_edge + PROBLEM.velocity_x * front_points[:, 2]
        left_y = PROBLEM.left_edge + PROBLEM.velocity_y * front_points[:, 2]
        right_y = PROBLEM.right_edge + PROBLEM.velocity_y * front_points[:, 2]

        front_points[:, 0] = front_points[:, 0] * PROBLEM.length
        front_points[:, 1] = front_points[:, 1] * PROBLEM.length
        jitter = (np.random.rand(n_front) - 0.5) * 2.0 * width

        x_left = selector == 0
        x_right = selector == 1
        y_left = selector == 2
        y_right = selector == 3
        front_points[x_left, 0] = left_x[x_left] + jitter[x_left]
        front_points[x_right, 0] = right_x[x_right] + jitter[x_right]
        front_points[y_left, 1] = left_y[y_left] + jitter[y_left]
        front_points[y_right, 1] = right_y[y_right] + jitter[y_right]
        front_points[:, 0] = np.clip(front_points[:, 0], 0.0, PROBLEM.length)
        front_points[:, 1] = np.clip(front_points[:, 1], 0.0, PROBLEM.length)

        points = np.vstack([random_points, initial_points, front_points])
        X_s = torch.tensor(points, dtype=DTYPE, device=DEVICE)
        phi_s = exact_solution(X_s[:, 0:1], X_s[:, 1:2], X_s[:, 2:3]).detach()
        return X_s, phi_s

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

    def adaptive_refine(self):
        X_cand = self.sample_f()
        residual = self.pde_residual(X_cand).abs().squeeze()
        _, idx = torch.topk(residual, TRAIN.n_adap)
        X_new = X_cand[idx].detach().requires_grad_(True)
        self.X_f = torch.cat([self.X_f, X_new], dim=0)

    def loss_weights(self):
        if self.step_id == 1:
            return (
                TRAIN.first_step_pde_weight,
                TRAIN.bc_weight,
                TRAIN.first_step_data_weight,
                TRAIN.first_step_range_weight,
                TRAIN.first_step_ic_weight,
            )
        return (
            TRAIN.pde_weight,
            TRAIN.bc_weight,
            TRAIN.data_weight,
            TRAIN.range_weight,
            TRAIN.ic_weight,
        )

    def range_loss(self):
        X_range = torch.cat([self.X_s, self.X_b, self.X_ic], dim=0)
        phi = self.model(X_range)
        below_zero = torch.relu(-phi)
        above_one = torch.relu(phi - 1.0)
        return torch.mean(below_zero**2 + above_one**2)

    def loss_components(self):
        loss_pde = torch.mean(self.pde_residual(self.X_f) ** 2)
        loss_bc = torch.mean((self.model(self.X_b) - self.phi_b) ** 2)
        loss_data = torch.mean((self.model(self.X_s) - self.phi_s) ** 2)
        loss_range = self.range_loss()
        loss_ic = torch.mean((self.model(self.X_ic) - self.phi_ic) ** 2)
        return loss_pde, loss_bc, loss_data, loss_range, loss_ic

    def loss(self):
        loss_pde, loss_bc, loss_data, loss_range, loss_ic = self.loss_components()
        pde_weight, bc_weight, data_weight, range_weight, ic_weight = self.loss_weights()
        total_loss = (
            pde_weight * loss_pde
            + bc_weight * loss_bc
            + data_weight * loss_data
            + range_weight * loss_range
            + ic_weight * loss_ic
        )
        self.last_loss_parts = (
            loss_pde.detach(),
            loss_bc.detach(),
            loss_data.detach(),
            loss_range.detach(),
            loss_ic.detach(),
            total_loss.detach(),
        )
        return total_loss

    def _log_and_step(self, loss):
        self.iter += 1
        if self.iter % TRAIN.print_every == 0:
            if self.last_loss_parts is None:
                print(f"Iter {self.iter}, loss = {loss.item():.3e}")
                return

            loss_pde, loss_bc, loss_data, loss_range, loss_ic, total_loss = (
                self.last_loss_parts
            )
            print(
                f"Iter {self.iter}, loss = {total_loss.item():.3e}, "
                f"pde = {loss_pde.item():.3e}, "
                f"bc = {loss_bc.item():.3e}, "
                f"data = {loss_data.item():.3e}, "
                f"range = {loss_range.item():.3e}, "
                f"ic = {loss_ic.item():.3e}"
            )

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
    n_random = TRAIN.n_i // 2
    n_edge = TRAIN.n_i // 5
    n_inside = TRAIN.n_i - n_random - 2 * n_edge

    random_points = lhs(2, n_random)
    random_points[:, 0] = random_points[:, 0] * PROBLEM.length
    random_points[:, 1] = random_points[:, 1] * PROBLEM.length

    y_span = lhs(1, n_edge)[:, 0] * PROBLEM.length
    x_span = lhs(1, n_edge)[:, 0] * PROBLEM.length
    x_edges = np.random.choice([PROBLEM.left_edge, PROBLEM.right_edge], size=n_edge)
    y_edges = np.random.choice([PROBLEM.left_edge, PROBLEM.right_edge], size=n_edge)
    edge_x_points = np.column_stack([x_edges, y_span])
    edge_y_points = np.column_stack([x_span, y_edges])

    inside_points = lhs(2, n_inside)
    inside_points[:, 0] = (
        PROBLEM.left_edge
        + inside_points[:, 0] * (PROBLEM.right_edge - PROBLEM.left_edge)
    )
    inside_points[:, 1] = (
        PROBLEM.left_edge
        + inside_points[:, 1] * (PROBLEM.right_edge - PROBLEM.left_edge)
    )

    points = np.vstack([random_points, edge_x_points, edge_y_points, inside_points])
    return torch.tensor(points, dtype=DTYPE, device=DEVICE)


def solve():
    xy_ic = sample_initial_points()
    phi_ic = initial_condition(xy_ic[:, 0:1], xy_ic[:, 1:2])

    prev_model = None
    total_start = time.time()

    for n in range(TRAIN.n_steps):
        t0 = PROBLEM.initial_time + n * TRAIN.dt
        t1 = t0 + TRAIN.dt
        print(f"\n=== Step {n + 1}/{TRAIN.n_steps}, t in [{t0:.6f}, {t1:.6f}] ===")

        pinn = SingleStepPINN(
            t0=t0,
            xy_ic=xy_ic,
            phi_ic=phi_ic,
            prev_model=prev_model,
            step_id=n + 1,
        )
        prev_model = pinn.train()

        t_next = torch.full_like(xy_ic[:, 0:1], t1)
        if TRAIN.use_exact_step_initial:
            phi_ic = exact_solution(xy_ic[:, 0:1], xy_ic[:, 1:2], t_next).detach()
        else:
            X_next = torch.cat([xy_ic[:, 0:1], xy_ic[:, 1:2], t_next], dim=1).to(DEVICE)
            with torch.no_grad():
                phi_ic = prev_model(X_next).detach()

    total_elapsed = time.time() - total_start
    print("\n==============================")
    print(f"Total training time = {total_elapsed:.2f} s")
    print("==============================")


if __name__ == "__main__":
    solve()

# Total training time = 1553.31 s
# 先做ppt，再降ai率
