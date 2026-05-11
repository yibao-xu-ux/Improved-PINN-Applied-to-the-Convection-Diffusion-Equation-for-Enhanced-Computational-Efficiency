import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import square_wave_2d_train as train_module


DEVICE = train_module.DEVICE
DTYPE = train_module.DTYPE
PROBLEM = train_module.PROBLEM
TRAIN = train_module.TRAIN
PINN = train_module.PINN

MODEL_DIR = TRAIN.save_dir
BASE_DIR = train_module.BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, "square_wave_2d_figures")
os.makedirs(FIG_DIR, exist_ok=True)


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
    return X, Y, pred.cpu().numpy(), exact.cpu().numpy()


def plot_field_triptychs():
    x = np.linspace(0.0, PROBLEM.length, 180)
    y = np.linspace(0.0, PROBLEM.length, 180)
    times = [PROBLEM.initial_time, 0.3, 0.6, TRAIN.t_end]

    for t in times:
        X, Y, pred, exact = predict_on_grid(x, y, t)
        abs_error = np.abs(pred - exact)

        fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.2), constrained_layout=True)
        field_min = min(float(pred.min()), float(exact.min()), 0.0)
        field_max = max(float(pred.max()), float(exact.max()), 1.0)
        field_levels = np.linspace(field_min, field_max, 44)
        error_levels = np.linspace(0.0, max(float(abs_error.max()), 1.0e-8), 44)

        panels = [
            (pred, "PINN solution", field_levels, "viridis"),
            (exact, "Exact solution", field_levels, "viridis"),
            (abs_error, "Absolute error", error_levels, "magma"),
        ]

        for ax, (values, title, levels, cmap) in zip(axes, panels):
            contour = ax.contourf(X, Y, values, levels=levels, cmap=cmap, extend="both")
            ax.set_title(title)
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(0.0, PROBLEM.length)
            ax.set_ylim(0.0, PROBLEM.length)
            cbar = fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.025)
            cbar.ax.tick_params(labelsize=9)

        fig.suptitle(rf"2D square wave at $t={t:.4f}$", y=1.04, fontsize=14)
        fig.savefig(os.path.join(FIG_DIR, f"field_triptych_t_{t:.4f}.png"))
        plt.close(fig)


def plot_t03_comparison():
    x = np.linspace(0.0, PROBLEM.length, 1000)
    y_center = 0.5 * (PROBLEM.left_edge + PROBLEM.right_edge)
    t = 0.3
    _, _, pred_2d, exact_2d = predict_on_grid(x, np.array([y_center]), t)
    pred = pred_2d[0]
    exact = exact_2d[0]

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.plot(x, exact, color="#1f4e79", lw=2.2, label="Reference")
    ax.plot(x, pred, color="#c43c39", lw=1.9, ls="--", label="PINN")
    ax.fill_between(x, exact, pred, color="#c43c39", alpha=0.13, linewidth=0)
    ax.set_title(rf"$y={y_center:.1f},\ t={t:.4f}$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\phi(x,y,t)$")
    ax.set_xlim(0.0, PROBLEM.length)
    ax.set_ylim(-0.12, 1.12)
    ax.grid(True, color="#d8d8d8", lw=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False)

    fig.savefig(os.path.join(FIG_DIR, "line_comparison_y_1.5_t_0.3000.png"))
    plt.close(fig)


def plot_centerline_profiles():
    x = np.linspace(0.0, PROBLEM.length, 800)
    y_center = 0.5 * (PROBLEM.left_edge + PROBLEM.right_edge)
    times = [PROBLEM.initial_time, 0.3, 0.6, TRAIN.t_end]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, t in zip(axes, times):
        _, _, pred_2d, exact_2d = predict_on_grid(x, np.array([y_center]), t)
        pred = pred_2d[0]
        exact = exact_2d[0]
        ax.plot(x, exact, color="#1f4e79", lw=2.2, label="Exact")
        ax.plot(x, pred, color="#c43c39", lw=1.9, ls="--", label="PINN")
        ax.fill_between(x, exact, pred, color="#c43c39", alpha=0.13, linewidth=0)
        ax.set_title(rf"$y={y_center:.1f},\ t={t:.4f}$")
        ax.set_xlim(0.0, PROBLEM.length)
        ax.set_ylim(-0.12, 1.12)
        ax.grid(True, color="#d8d8d8", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel(r"$x$")
    for ax in axes[::2]:
        ax.set_ylabel(r"$\phi(x,y,t)$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(os.path.join(FIG_DIR, "centerline_profiles.png"))
    plt.close(fig)


def plot_surface_comparison():
    x = np.linspace(0.0, PROBLEM.length, 90)
    y = np.linspace(0.0, PROBLEM.length, 90)
    t = TRAIN.t_end
    X, Y, pred, exact = predict_on_grid(x, y, t)
    abs_error = np.abs(pred - exact)

    fig = plt.figure(figsize=(15.0, 4.8), constrained_layout=True)
    panels = [
        (pred, "PINN solution", "viridis"),
        (exact, "Exact solution", "viridis"),
        (abs_error, "Absolute error", "magma"),
    ]

    for i, (values, title, cmap) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        surface = ax.plot_surface(
            X,
            Y,
            values,
            cmap=cmap,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$\phi$")
        ax.set_xlim(0.0, PROBLEM.length)
        ax.set_ylim(0.0, PROBLEM.length)
        ax.view_init(elev=28, azim=-128)
        fig.colorbar(surface, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle(rf"Surface comparison at $t={t:.2f}$", y=1.03, fontsize=14)
    fig.savefig(os.path.join(FIG_DIR, "surface_comparison_t_end.png"))
    plt.close(fig)


def main():
    configure_matplotlib()
    plot_t03_comparison()
    print(f"Figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
