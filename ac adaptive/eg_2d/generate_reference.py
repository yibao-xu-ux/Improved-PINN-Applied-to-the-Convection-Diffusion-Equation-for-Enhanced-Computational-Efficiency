import argparse
import os

import numpy as np

from train import DT, EPS, LAMBDA, N_STEPS


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SAVE_PATH = os.path.join(SCRIPT_DIR, "reference_2d.npz")


def initial_condition(x, y):
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    return np.tanh((0.35 - r) / (2.0 * EPS))


def solve_reference(n=256, dt=1.0e-3, t_final=None, num_times=101):
    if t_final is None:
        t_final = DT * N_STEPS

    n = int(n)
    dt = float(dt)
    t_final = float(t_final)
    save_times = np.linspace(0.0, t_final, int(num_times))

    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")
    u = initial_condition(x_grid, y_grid).astype(np.float64)

    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    k2 = kx**2 + ky**2
    denom = 1.0 + dt * LAMBDA * EPS**2 * k2

    uu = np.empty((len(save_times), n, n), dtype=np.float32)
    uu[0] = u.astype(np.float32)

    t = 0.0
    save_id = 1
    n_steps = int(np.ceil(t_final / dt))

    for step in range(1, n_steps + 1):
        dt_step = min(dt, t_final - t)
        if dt_step <= 0.0:
            break

        denom_step = denom
        if dt_step != dt:
            denom_step = 1.0 + dt_step * LAMBDA * EPS**2 * k2

        rhs = u + dt_step * LAMBDA * (u - u**3)
        u = np.fft.ifft2(np.fft.fft2(rhs) / denom_step).real
        t += dt_step

        while save_id < len(save_times) and t + 1.0e-12 >= save_times[save_id]:
            uu[save_id] = u.astype(np.float32)
            save_id += 1

        if step % max(1, n_steps // 20) == 0 or step == n_steps:
            print(f"step {step:6d}/{n_steps}, t = {t:.6f}")

    return {
        "x": x.astype(np.float64),
        "y": y.astype(np.float64),
        "tt": save_times.astype(np.float64),
        "uu": uu,
        "n": np.array(n, dtype=np.int32),
        "dt": np.array(dt, dtype=np.float64),
        "eps": np.array(EPS, dtype=np.float64),
        "lambda": np.array(LAMBDA, dtype=np.float64),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2D Allen-Cahn numerical reference solution."
    )
    parser.add_argument("--n", type=int, default=256, help="grid size in x and y")
    parser.add_argument("--dt", type=float, default=1.0e-3, help="reference time step")
    parser.add_argument(
        "--num-times",
        type=int,
        default=101,
        help="number of saved output times including endpoints",
    )
    parser.add_argument("--t-final", type=float, default=DT * N_STEPS, help="final time")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH, help="output .npz path")
    args = parser.parse_args()

    data = solve_reference(
        n=args.n,
        dt=args.dt,
        t_final=args.t_final,
        num_times=args.num_times,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    np.savez_compressed(args.save_path, **data)
    print(f"\nreference solution saved to: {args.save_path}")


if __name__ == "__main__":
    main()
