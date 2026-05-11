import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import square_wave_train as train_module


DEVICE = train_module.DEVICE
DTYPE = train_module.DTYPE
PROBLEM = train_module.PROBLEM
TRAIN = train_module.TRAIN
PINN = train_module.PINN

MODEL_DIR = TRAIN.save_dir
BASE_DIR = train_module.BASE_DIR
FIG_DIR = os.path.join(BASE_DIR, "square_wave_figures")
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


def exact_values(x_tensor, t_tensor):
    if float(t_tensor[0].item()) == 0.0:
        return train_module.initial_condition(x_tensor).squeeze(1)
    return train_module.exact_solution(x_tensor, t_tensor).squeeze(1)


def predict_at_time(x, t):
    x_tensor = torch.as_tensor(x, dtype=DTYPE, device=DEVICE).view(-1, 1)
    t_tensor = torch.full_like(x_tensor, float(t))
    inputs = torch.cat([x_tensor, t_tensor], dim=1)
    model = load_model(step_id_from_time(t))
    with torch.no_grad():
        pred = model(inputs).squeeze(1)
        exact = exact_values(x_tensor, t_tensor)
    return pred.cpu().numpy(), exact.cpu().numpy()


def predict_on_grid(x_values, t_values):
    X, T = np.meshgrid(x_values, t_values)
    pred = np.empty_like(X, dtype=np.float64)
    exact = np.empty_like(X, dtype=np.float64)

    for i, t in enumerate(t_values):
        pred[i], exact[i] = predict_at_time(x_values, t)

    return X, T, pred, exact


def plot_line_profiles():
    x = np.linspace(0.0, PROBLEM.length, 1000)
    times = [0.0, 0.2, 0.4, 0.6, 0.8, TRAIN.t_end]

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, t in zip(axes, times):
        pred, exact = predict_at_time(x, t)
        ax.plot(x, exact, color="#1f4e79", lw=2.2, label="Exact")
        ax.plot(x, pred, color="#c43c39", lw=1.9, ls="--", label="PINN")
        ax.fill_between(x, exact, pred, color="#c43c39", alpha=0.13, linewidth=0)
        ax.set_title(rf"$t={t:.2f}$")
        ax.set_xlim(0.0, PROBLEM.length)
        ax.set_ylim(-0.12, 1.12)
        ax.grid(True, color="#d8d8d8", lw=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[3:]:
        ax.set_xlabel(r"$x$")
    for ax in axes[::3]:
        ax.set_ylabel(r"$\phi(x,t)$")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(os.path.join(FIG_DIR, "line_profiles_pinn_vs_exact.png"))
    plt.close(fig)


def plot_spacetime_fields():
    x = np.linspace(0.0, PROBLEM.length, 420)
    t = np.linspace(0.0, TRAIN.t_end, 240)
    X, T, pred, exact = predict_on_grid(x, t)
    abs_error = np.abs(pred - exact)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1), constrained_layout=True)
    field_min = min(float(pred.min()), float(exact.min()), 0.0)
    field_max = max(float(pred.max()), float(exact.max()), 1.0)
    field_levels = np.linspace(field_min, field_max, 42)
    error_levels = np.linspace(0.0, max(abs_error.max(), 1.0e-8), 42)

    panels = [
        (pred, "PINN solution", field_levels, "viridis"),
        (exact, "Exact solution", field_levels, "viridis"),
        (abs_error, "Absolute error", error_levels, "magma"),
    ]

    for ax, (values, title, levels, cmap) in zip(axes, panels):
        contour = ax.contourf(X, T, values, levels=levels, cmap=cmap, extend="both")
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$t$")
        ax.set_xlim(0.0, PROBLEM.length)
        ax.set_ylim(0.0, TRAIN.t_end)
        cbar = fig.colorbar(contour, ax=ax, fraction=0.048, pad=0.025)
        cbar.ax.tick_params(labelsize=9)

    fig.savefig(os.path.join(FIG_DIR, "spacetime_solution_and_error.png"))
    plt.close(fig)


def main():
    configure_matplotlib()
    plot_line_profiles()
    plot_spacetime_fields()
    print(f"Figures saved to: {os.path.abspath(FIG_DIR)}")


if __name__ == "__main__":
    main()
