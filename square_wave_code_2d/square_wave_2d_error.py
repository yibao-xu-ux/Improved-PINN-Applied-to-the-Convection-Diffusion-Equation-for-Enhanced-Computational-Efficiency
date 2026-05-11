import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import square_wave_2d_train as train_module


DEVICE = train_module.DEVICE
DTYPE = train_module.DTYPE
PROBLEM = train_module.PROBLEM
TRAIN = train_module.TRAIN
PINN = train_module.PINN

MODEL_DIR = TRAIN.save_dir
BASE_DIR = train_module.BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, "square_wave_2d_figures")
REPORT_DIR = os.path.join(BASE_DIR, "square_wave_2d_reports")
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


def step_id_from_time(t):
    step_id = int(np.floor((float(t) - PROBLEM.initial_time) / TRAIN.dt)) + 1
    return min(max(step_id, 1), TRAIN.n_steps)


_MODEL_CACHE = {}


def load_model(step_id):
    if step_id not in _MODEL_CACHE:
        model = PINN().to(DEVICE)
        model.set_step_start_time(PROBLEM.initial_time + (step_id - 1) * TRAIN.dt)
        path = os.path.join(MODEL_DIR, f"step_{step_id:03d}.pth")
        state = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE[step_id] = model
    return _MODEL_CACHE[step_id]


def predict_on_grid(x_values, y_values, t):
    X, Y = np.meshgrid(x_values, y_values)
    x_tensor = torch.as_tensor(X.ravel(), dtype=DTYPE, device=DEVICE).view(-1, 1)
    y_tensor = torch.as_tensor(Y.ravel(), dtype=DTYPE, device=DEVICE).view(-1, 1)
    t_tensor = torch.full_like(x_tensor, float(t))
    inputs = torch.cat([x_tensor, y_tensor, t_tensor], dim=1)

    model = load_model(step_id_from_time(t))
    with torch.no_grad():
        pred = model(inputs).view(len(y_values), len(x_values))
        exact = train_module.exact_solution(x_tensor, y_tensor, t_tensor).view(
            len(y_values),
            len(x_values),
        )
    return pred.cpu().numpy(), exact.cpu().numpy()


def evaluate_time_sequence(nx=160, ny=160, nt=61):
    x_values = np.linspace(0.0, PROBLEM.length, nx)
    y_values = np.linspace(0.0, PROBLEM.length, ny)
    t_values = np.linspace(PROBLEM.initial_time, TRAIN.t_end, nt)

    relative_l2 = np.empty(nt, dtype=np.float64)
    max_abs = np.empty(nt, dtype=np.float64)

    sum_error_l2 = 0.0
    sum_exact_l2 = 0.0
    sum_error_l1 = 0.0
    sum_exact_l1 = 0.0
    sum_error_sq = 0.0
    sum_abs_error = 0.0
    total_count = 0
    global_max_abs = 0.0

    for i, t in enumerate(t_values):
        pred, exact = predict_on_grid(x_values, y_values, t)
        error = pred - exact
        abs_error = np.abs(error)

        relative_l2[i] = np.linalg.norm(error.ravel(), ord=2) / np.linalg.norm(
            exact.ravel(),
            ord=2,
        )
        max_abs[i] = abs_error.max()

        sum_error_l2 += np.sum(error**2)
        sum_exact_l2 += np.sum(exact**2)
        sum_error_l1 += np.sum(abs_error)
        sum_exact_l1 += np.sum(np.abs(exact))
        sum_error_sq += np.sum(error**2)
        sum_abs_error += np.sum(abs_error)
        total_count += error.size
        global_max_abs = max(global_max_abs, float(abs_error.max()))

    metrics = {
        "global_relative_l2": np.sqrt(sum_error_l2) / np.sqrt(sum_exact_l2),
        "global_relative_l1": sum_error_l1 / sum_exact_l1,
        "rmse": np.sqrt(sum_error_sq / total_count),
        "mae": sum_abs_error / total_count,
        "max_absolute_error": global_max_abs,
    }

    return t_values, relative_l2, max_abs, metrics


def save_metrics(metrics, nx, ny, nt):
    path = os.path.join(REPORT_DIR, "error_metrics.txt")
    with open(path, "w", encoding="utf-8") as file:
        file.write("2D square wave PINN error metrics\n")
        file.write("=================================\n")
        file.write(f"Model directory: {MODEL_DIR}\n")
        file.write(f"Domain: x,y in [0, {PROBLEM.length:.12f}], t in [{PROBLEM.initial_time}, {TRAIN.t_end}]\n")
        file.write(f"Evaluation grid: nx={nx}, ny={ny}, nt={nt}\n")
        file.write("\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.8e}\n")
    return path


def plot_time_error(t_values, relative_l2, max_abs):
    fig, ax_l2 = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    ax_abs = ax_l2.twinx()

    line_l2 = ax_l2.plot(
        t_values,
        relative_l2,
        color="#1f4e79",
        lw=2.2,
        label=r"Relative $L^2$ error",
    )
    line_abs = ax_abs.plot(
        t_values,
        max_abs,
        color="#c43c39",
        lw=2.0,
        ls="--",
        label="Max absolute error",
    )

    ax_l2.set_title("2D square wave error evolution")
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


def plot_final_error_heatmap():
    x = np.linspace(0.0, PROBLEM.length, 220)
    y = np.linspace(0.0, PROBLEM.length, 220)
    X, Y = np.meshgrid(x, y)
    pred, exact = predict_on_grid(x, y, TRAIN.t_end)
    abs_error = np.abs(pred - exact)

    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    levels = np.linspace(0.0, max(float(abs_error.max()), 1.0e-8), 50)
    contour = ax.contourf(X, Y, abs_error, levels=levels, cmap="magma", extend="max")
    ax.set_title(rf"Absolute error at $t={TRAIN.t_end:.2f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_xlim(0.0, PROBLEM.length)
    ax.set_ylim(0.0, PROBLEM.length)
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025)
    cbar.set_label(r"$|\phi_{\mathrm{PINN}}-\phi_{\mathrm{exact}}|$")
    fig.savefig(os.path.join(FIG_DIR, "final_absolute_error_heatmap.png"))
    plt.close(fig)


def main():
    configure_matplotlib()
    nx, ny, nt = 160, 160, 61
    t_values, relative_l2, max_abs, metrics = evaluate_time_sequence(nx, ny, nt)
    report_path = save_metrics(metrics, nx, ny, nt)
    plot_time_error(t_values, relative_l2, max_abs)
    plot_final_error_heatmap()

    print("Error metrics")
    print("=============")
    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")
    print(f"\nReport saved to: {os.path.abspath(report_path)}")
    print(f"Figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
