import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from train import DEVICE, DT, DTYPE, MODEL_DIR, N_LAYERS, N_NEURONS, N_STEPS, Network


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(SCRIPT_DIR, "reference_2d.npz")


def get_step_id(t):
    step = int(np.floor(float(t) / DT)) + 1
    return min(step, N_STEPS)


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_model_2d(step_id, folder=MODEL_DIR):
    model = Network(3, N_NEURONS, 1, N_LAYERS).to(DEVICE)
    candidates = [
        os.path.join(folder, f"step_{step_id:03d}.pth"),
        os.path.join(SCRIPT_DIR, folder, f"step_{step_id:03d}.pth"),
        os.path.join(os.path.dirname(SCRIPT_DIR), folder, f"step_{step_id:03d}.pth"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"model file not found for step {step_id}: {path}")

    model.load_state_dict(_torch_load(path))
    model.eval()
    return model


def evaluate_pinn(model, x_grid, y_grid, t_value, batch_size=65536):
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    u_flat = np.empty(x_flat.shape, dtype=np.float32)

    for start in range(0, x_flat.size, batch_size):
        end = min(start + batch_size, x_flat.size)
        x_t = torch.tensor(x_flat[start:end], dtype=DTYPE, device=DEVICE).view(-1, 1)
        y_t = torch.tensor(y_flat[start:end], dtype=DTYPE, device=DEVICE).view(-1, 1)
        t_t = torch.full_like(x_t, float(t_value))

        with torch.no_grad():
            u = model(torch.cat([x_t, y_t, t_t], dim=1))
        u_flat[start:end] = u.cpu().numpy().squeeze()

    return u_flat.reshape(x_grid.shape)


def load_reference(path=REFERENCE_PATH):
    if not os.path.exists(path):
        return None
    return np.load(path)


def nearest_reference(data, t_value):
    tt = data["tt"]
    j = int(np.argmin(np.abs(tt - float(t_value))))
    return float(tt[j]), np.asarray(data["uu"][j])


def plot_pinn_only(t_plot_list):
    n = 200
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    u_list = []
    for t_plot in t_plot_list:
        model = load_model_2d(get_step_id(t_plot))
        u_list.append(evaluate_pinn(model, x_grid, y_grid, t_plot))

    vmin = min(np.min(u) for u in u_list)
    vmax = max(np.max(u) for u in u_list)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes = axes.flatten()
    im = None

    for ax, u, t_plot in zip(axes, u_list, t_plot_list):
        im = ax.imshow(
            u,
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"PINN, t = {t_plot:g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, location="right")
    cbar.set_label(r"$u(x,y,t)$")
    return fig, "plot_2D.png"


def plot_comparison(t_plot_list, ref):
    x = ref["x"]
    y = ref["y"]
    x_grid, y_grid = np.meshgrid(x, y, indexing="xy")

    pinn_list = []
    ref_list = []
    err_list = []
    rel_l2_list = []

    for t_plot in t_plot_list:
        t_eval, u_ref = nearest_reference(ref, t_plot)
        model = load_model_2d(get_step_id(t_eval))
        u_pinn = evaluate_pinn(model, x_grid, y_grid, t_eval)
        err = np.abs(u_pinn - u_ref)
        rel_l2 = np.linalg.norm(u_pinn - u_ref) / np.linalg.norm(u_ref)

        pinn_list.append(u_pinn)
        ref_list.append(u_ref)
        err_list.append(err)
        rel_l2_list.append(rel_l2)

    vmin = min(np.min(u) for u in pinn_list + ref_list)
    vmax = max(np.max(u) for u in pinn_list + ref_list)
    err_vmax = max(np.max(e) for e in err_list)

    fig, axes = plt.subplots(3, 6, figsize=(18, 8), constrained_layout=True)
    im_solution = None
    im_error = None

    for col, t_plot in enumerate(t_plot_list):
        im_solution = axes[0, col].imshow(
            ref_list[col],
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, col].set_title(f"Reference, t = {t_plot:g}")

        axes[1, col].imshow(
            pinn_list[col],
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="RdBu_r",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, col].set_title(f"PINN, t = {t_plot:g}")

        im_error = axes[2, col].imshow(
            err_list[col],
            extent=[0, 1, 0, 1],
            origin="lower",
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=err_vmax,
        )
        axes[2, col].set_title(f"|error|, rel L2={rel_l2_list[col]:.2e}")

    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    cbar_u = fig.colorbar(im_solution, ax=axes[:2, :], shrink=0.9, location="right")
    cbar_u.set_label(r"$u(x,y,t)$")
    cbar_e = fig.colorbar(im_error, ax=axes[2, :], shrink=0.9, location="right")
    cbar_e.set_label(r"$|u_{PINN}-u_{ref}|$")
    return fig, "plot_2D_compare.png"


if __name__ == "__main__":
    t_plot_list = [0, 2, 4, 6, 8, 10]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 11,
        }
    )

    ref = load_reference()
    if ref is None:
        fig, save_name = plot_pinn_only(t_plot_list)
        print(
            "\nreference_2d.npz not found; generated PINN-only plot. "
            "Run generate_reference.py first for comparison plots."
        )
    else:
        fig, save_name = plot_comparison(t_plot_list, ref)

    save_path = os.path.join(SCRIPT_DIR, save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nfigure saved to: {save_path}")
    plt.show()
