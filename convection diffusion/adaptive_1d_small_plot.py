from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models_stm_small"

DT = 0.1
T_END = 1.0
N_STEPS = int(round(T_END / DT))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
N_LAYERS = 4
N_NEURONS = 40


class PINN(nn.Module):
    """Network architecture matching adaptive_1d_small.PINN."""

    def __init__(self):
        super().__init__()
        layers = [nn.Linear(2, N_NEURONS), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(N_NEURONS, N_NEURONS), nn.Tanh()]
        layers.append(nn.Linear(N_NEURONS, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def analytical_solution(x, t, a=1.0, kappa=0.1 / np.pi):
    """Reference solution for the 1D convection-diffusion equation."""
    n_terms = 800
    x = np.asarray(x)
    t1 = np.sinh(a / (2 * kappa))
    t2 = np.cosh(a / (2 * kappa))
    part1 = np.zeros_like(x, dtype=np.float64)
    part2 = np.zeros_like(x, dtype=np.float64)

    for p in range(n_terms + 1):
        term_1_num = (
            (-1) ** p
            * (2 * p)
            * np.sin(np.pi * p * x)
            * np.exp(-kappa * p**2 * np.pi**2 * t)
        )
        term_1_den = (
            a**4
            + 8 * (a * kappa * np.pi) ** 2 * (p**2 + 1)
            + 16 * (np.pi * kappa) ** 4 * (p**2 - 1) ** 2
        )

        q = 2 * p + 1
        term_2_num = (
            (-1) ** p
            * q
            * np.cos(q / 2 * np.pi * x)
            * np.exp(-kappa * q**2 / 4 * np.pi**2 * t)
        )
        term_2_den = (
            a**4
            + (a * kappa * np.pi) ** 2 * (8 * p**2 + 8 * p + 10)
            + (np.pi * kappa) ** 4 * (4 * p**2 + 4 * p - 3) ** 2
        )

        part1 += term_1_num / term_1_den
        part2 += term_2_num / term_2_den

    result = t1 * part1 + t2 * part2
    result *= (
        16
        * np.pi**2
        * kappa**3
        * a
        * np.exp((a / (2 * kappa)) * (x - a * t / 2))
    )
    return result


_MODEL_CACHE = {}


def get_step_id(t):
    """Use the model trained for the time slab ending at t."""
    return int(np.clip(round(t / DT), 1, N_STEPS))


def load_model(step_id):
    if step_id not in _MODEL_CACHE:
        model_path = MODEL_DIR / f"step_{step_id:03d}.pth"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        model = PINN().to(DEVICE)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        _MODEL_CACHE[step_id] = model
    return _MODEL_CACHE[step_id]


def predict(model, x_ref, t_val):
    x_torch = torch.tensor(x_ref, dtype=DTYPE, device=DEVICE).view(-1, 1)
    t_torch = torch.full_like(x_torch, t_val)
    with torch.no_grad():
        return model(torch.cat([x_torch, t_torch], dim=1)).cpu().numpy().squeeze()


def set_plot_style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main():
    set_plot_style()

    t_plot_list = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    x_ref = np.linspace(-1.0, 1.0, 500)

    snapshots = []
    for t_val in t_plot_list:
        step_id = get_step_id(t_val)
        model = load_model(step_id)
        u_pred = predict(model, x_ref, t_val)
        u_ref = analytical_solution(x_ref, t_val)
        abs_err = np.abs(u_pred - u_ref)
        snapshots.append((t_val, step_id, u_pred, u_ref, abs_err))

    all_u = np.concatenate([s[2] for s in snapshots] + [s[3] for s in snapshots])
    y_pad = 0.08 * max(np.ptp(all_u), 1e-8)
    y_lim = (all_u.min() - y_pad, all_u.max() + y_pad)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.5, 7.2),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    pinn_color = "#d62728"
    ref_color = "#1f77b4"
    err_color = "#f2a65a"

    for ax, (t_val, step_id, u_pred, u_ref, abs_err) in zip(axes, snapshots):
        ax.plot(x_ref, u_ref, color=ref_color, lw=2.3, label="Reference")
        ax.plot(x_ref, u_pred, color=pinn_color, lw=1.9, ls="--", label="PINN")
        ax.fill_between(
            x_ref,
            u_ref,
            u_pred,
            color=err_color,
            alpha=0.16,
            linewidth=0,
            label="Difference",
        )

        rel_l2 = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
        max_err = abs_err.max()
        ax.text(
            0.03,
            0.06,
            f"rel L2={rel_l2:.2e}\nmax err={max_err:.2e}",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "alpha": 0.88,
            },
        )

        ax.set_title(f"t = {t_val:.1f}   step {step_id:02d}", pad=8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(*y_lim)
        ax.grid(True, color="#d9d9d9", lw=0.7, alpha=0.65)
        ax.axhline(0.0, color="#777777", lw=0.7, alpha=0.45)

    for ax in axes[::3]:
        ax.set_ylabel("u(x,t)")
    for ax in axes[-3:]:
        ax.set_xlabel("x")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle("PINN Prediction vs Reference Solution", fontsize=16, y=1.08)

    png_path = BASE_DIR / "1d_true_vs_pinn.png"
    pdf_path = BASE_DIR / "1d_true_vs_pinn.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
