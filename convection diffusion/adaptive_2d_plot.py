import os

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import torch
import torch.nn as nn


# =====================================================
# Configuration
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# Must match adaptive_2d.py.
PHI = np.deg2rad(22.5)
A_VEC = np.array([np.cos(PHI), np.sin(PHI)])
KAPPA = 0.005
N_LAYERS = 4
N_NEURONS = 40

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_stm_2d_pe200")
OUT_PNG = os.path.join(BASE_DIR, "adaptive_2d_snapshots_3x3.png")
OUT_PDF = os.path.join(BASE_DIR, "adaptive_2d_snapshots_3x3.pdf")

STEP_TO_TIME = {
    1: 0.1,
    5: 0.5,
    6: 0.6,
}

N_GRID = 200


# =====================================================
# PINN architecture
# =====================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.Linear(3, N_NEURONS), nn.Tanh()]
        for _ in range(N_LAYERS - 1):
            layers += [nn.Linear(N_NEURONS, N_NEURONS), nn.Tanh()]
        layers.append(nn.Linear(N_NEURONS, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================
# Exact solution
# =====================================================
def exact_solution_np(X, Y, t):
    ax = A_VEC[0] * t
    ay = A_VEC[1] * t
    return (1.0 / (4.0 * t + 1.0)) * np.exp(
        -((X - ax) ** 2 + (Y - ay) ** 2) / (KAPPA * (4.0 * t + 1.0))
    )


# =====================================================
# Model and snapshot evaluation
# =====================================================
def load_model(step_id):
    model = PINN().to(DEVICE)
    state_path = os.path.join(MODEL_DIR, f"step_{step_id:03d}.pth")
    state = torch.load(state_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate_snapshot(model, t):
    x = np.linspace(0.0, 1.0, N_GRID)
    y = np.linspace(0.0, 1.0, N_GRID)
    X, Y = np.meshgrid(x, y, indexing="ij")
    T = np.full_like(X, t)

    XYT = np.stack([X, Y, T], axis=-1).reshape(-1, 3)
    XYT = torch.tensor(XYT, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        u_pred = model(XYT).cpu().numpy().reshape(N_GRID, N_GRID)

    u_exact = exact_solution_np(X, Y, t)
    abs_error = np.abs(u_pred - u_exact)
    return u_pred, u_exact, abs_error


# =====================================================
# Plotting
# =====================================================
def add_panel(ax, data, title, norm, cmap, show_ylabel=False):
    im = ax.imshow(
        data.T,
        extent=[0, 1, 0, 1],
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="equal",
    )
    ax.set_title(title, fontsize=10, pad=5)
    ax.set_xlabel(r"$x$", fontsize=9)
    if show_ylabel:
        ax.set_ylabel(r"$y$", fontsize=9)
    else:
        ax.set_yticklabels([])

    ax.tick_params(axis="both", labelsize=7, direction="in", length=3, width=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return im


def plot_combined(snapshots):
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 9,
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.7,
        "savefig.bbox": "tight",
    })

    times = [item["t"] for item in snapshots]
    pred_list = [item["pred"] for item in snapshots]
    exact_list = [item["exact"] for item in snapshots]
    error_list = [item["error"] for item in snapshots]

    u_min = min(arr.min() for arr in pred_list + exact_list)
    u_max = max(arr.max() for arr in pred_list + exact_list)
    u_norm = colors.Normalize(vmin=u_min, vmax=u_max)

    err_max = max(arr.max() for arr in error_list)
    err_norm = colors.Normalize(vmin=0.0, vmax=err_max)

    fig, axes = plt.subplots(3, 3, figsize=(9.6, 8.0), constrained_layout=True)

    row_labels = ["PINN solution", "Exact solution", "Absolute error"]
    row_data = [pred_list, exact_list, error_list]
    row_norms = [u_norm, u_norm, err_norm]
    row_cmaps = ["viridis", "viridis", "magma"]

    for row in range(3):
        last_im = None
        for col, t in enumerate(times):
            last_im = add_panel(
                axes[row, col],
                row_data[row][col],
                rf"$t={t:.2f}$",
                row_norms[row],
                row_cmaps[row],
                show_ylabel=(col == 0),
            )
            if col == 0:
                axes[row, col].text(
                    -0.28,
                    0.5,
                    row_labels[row],
                    transform=axes[row, col].transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                )

        cbar = fig.colorbar(
            last_im,
            ax=axes[row, :],
            location="right",
            fraction=0.025,
            pad=0.018,
        )
        cbar.ax.tick_params(labelsize=7, direction="in", length=3, width=0.7)
        if row == 2:
            cbar.formatter.set_powerlimits((0, 0))
            cbar.update_ticks()

    fig.savefig(OUT_PNG, dpi=400)
    fig.savefig(OUT_PDF)
    plt.close(fig)


def main():
    snapshots = []
    for step_id, t in STEP_TO_TIME.items():
        model = load_model(step_id)
        u_pred, u_exact, abs_error = evaluate_snapshot(model, t)
        snapshots.append({
            "step_id": step_id,
            "t": t,
            "pred": u_pred,
            "exact": u_exact,
            "error": abs_error,
        })
        print(f"Evaluated step {step_id:03d}, t={t:.2f}")

    plot_combined(snapshots)
    print(f"Saved combined figure:\n  {OUT_PNG}\n  {OUT_PDF}")


if __name__ == "__main__":
    main()
