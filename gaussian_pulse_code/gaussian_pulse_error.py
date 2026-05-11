import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import gaussian_pulse_train as train_module

DEVICE = train_module.DEVICE
DTYPE = train_module.DTYPE
PROBLEM = train_module.PROBLEM
TRAIN = train_module.TRAIN
PINN = train_module.PINN

MODEL_DIR = TRAIN.save_dir
BASE_DIR = train_module.BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, "gaussian_pulse_figures")
REPORT_DIR = os.path.join(BASE_DIR, "gaussian_pulse_reports")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.1,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def analytical_solution(x, t):
    denominator = PROBLEM.sigma**2 + 2.0 * PROBLEM.diffusion * t
    amplitude = PROBLEM.sigma / torch.sqrt(denominator)
    exponent = -((x - PROBLEM.velocity * t) ** 2) / (
        4.0 * PROBLEM.diffusion * t + 2.0 * PROBLEM.sigma**2
    )
    return amplitude * torch.exp(exponent)


def step_id_from_time(t):
    step_id = int(np.floor(float(t) / TRAIN.dt)) + 1
    return min(max(step_id, 1), TRAIN.n_steps)


_MODEL_CACHE = {}


def load_model(step_id):
    if step_id not in _MODEL_CACHE:
        model = PINN().to(DEVICE)
        path = os.path.join(MODEL_DIR, f"step_{step_id:03d}.pth")
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE[step_id] = model
    return _MODEL_CACHE[step_id]


def predict_at_time(x_values, t):
    x = torch.as_tensor(x_values, dtype=DTYPE, device=DEVICE).view(-1, 1)
    time = torch.full_like(x, float(t))
    inputs = torch.cat([x, time], dim=1)
    model = load_model(step_id_from_time(t))
    with torch.no_grad():
        pred = model(inputs).squeeze(1)
        exact = analytical_solution(x, time).squeeze(1)
    return pred.cpu().numpy(), exact.cpu().numpy()


def evaluate_on_grid(nx=500, nt=241):
    x_values = np.linspace(0.0, PROBLEM.length, nx)
    t_values = np.linspace(0.0, TRAIN.t_end, nt)

    pred = np.empty((nt, nx), dtype=np.float64)
    exact = np.empty((nt, nx), dtype=np.float64)

    for i, t in enumerate(t_values):
        pred[i], exact[i] = predict_at_time(x_values, t)

    return x_values, t_values, pred, exact


def compute_error_metrics(pred, exact):
    error = pred - exact
    abs_error = np.abs(error)

    global_relative_l2 = np.linalg.norm(error.ravel(), ord=2) / np.linalg.norm(
        exact.ravel(), ord=2
    )
    global_relative_l1 = np.linalg.norm(error.ravel(), ord=1) / np.linalg.norm(
        exact.ravel(), ord=1
    )
    relative_linf = abs_error.max() / np.abs(exact).max()
    rmse = np.sqrt(np.mean(error**2))
    mae = np.mean(abs_error)

    return {
        "global_relative_l2": global_relative_l2,
        "global_relative_l1": global_relative_l1,
        "relative_linf": relative_linf,
        "rmse": rmse,
        "mae": mae,
        "max_absolute_error": abs_error.max(),
    }


def compute_time_history(pred, exact):
    error = pred - exact
    time_relative_l2 = np.linalg.norm(error, axis=1) / np.linalg.norm(exact, axis=1)
    time_max_abs = np.max(np.abs(error), axis=1)
    return time_relative_l2, time_max_abs


def save_metrics(metrics):
    path = os.path.join(REPORT_DIR, "error_metrics.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write("Gaussian pulse PINN error metrics\n")
        file.write("=================================\n")
        file.write(f"Model directory: {MODEL_DIR}\n")
        file.write(f"Domain: x in [0, {PROBLEM.length:.12f}], t in [0, {TRAIN.t_end}]\n")
        file.write("\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.8e}\n")
    return path


def plot_time_error(t_values, time_relative_l2, time_max_abs):
    fig, ax_l2 = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax_abs = ax_l2.twinx()

    line_l2 = ax_l2.plot(
        t_values,
        time_relative_l2,
        color="#1f4e79",
        lw=2.2,
        label=r"Relative $L^2$ error",
    )
    line_abs = ax_abs.plot(
        t_values,
        time_max_abs,
        color="#c43c39",
        lw=2.0,
        ls="--",
        label="Max absolute error",
    )

    ax_l2.set_title("Error evolution in time")
    ax_l2.set_xlabel(r"$t$")
    ax_l2.set_ylabel(r"Relative $L^2$ error")
    ax_abs.set_ylabel("Max absolute error")
    ax_l2.grid(True, color="#d8d8d8", lw=0.7, alpha=0.75)
    ax_l2.spines["top"].set_visible(False)
    ax_abs.spines["top"].set_visible(False)

    lines = line_l2 + line_abs
    labels = [line.get_label() for line in lines]
    ax_l2.legend(lines, labels, loc="upper left", frameon=False)

    fig.savefig(os.path.join(FIG_DIR, "time_error_evolution.png"))
    plt.close(fig)


def main():
    configure_matplotlib()
    x_values, t_values, pred, exact = evaluate_on_grid()
    metrics = compute_error_metrics(pred, exact)
    time_relative_l2, time_max_abs = compute_time_history(pred, exact)

    report_path = save_metrics(metrics)
    plot_time_error(t_values, time_relative_l2, time_max_abs)

    print("Error metrics")
    print("=============")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")
    print(f"\nReport saved to: {os.path.abspath(report_path)}")
    print(f"Figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
