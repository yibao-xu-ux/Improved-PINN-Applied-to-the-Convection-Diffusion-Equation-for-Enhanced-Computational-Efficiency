import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
from skimage import measure

from train import DEVICE, DT, DTYPE, MODEL_DIR, N_LAYERS, N_NEURONS, N_STEPS, Network


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATH = os.path.join(SCRIPT_DIR, "reference_3d.npz")


def get_step_id(t):
    step = int(np.floor(float(t) / DT)) + 1
    return min(step, N_STEPS)


def _torch_load(path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def load_model_3d(step_id, folder=MODEL_DIR):
    model = Network(4, N_NEURONS, 1, N_LAYERS).to(DEVICE)
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


def evaluate_pinn(model, x_grid, y_grid, z_grid, t_value, batch_size=65536):
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    z_flat = z_grid.reshape(-1)
    u_flat = np.empty(x_flat.shape, dtype=np.float32)

    for start in range(0, x_flat.size, batch_size):
        end = min(start + batch_size, x_flat.size)
        x_t = torch.tensor(x_flat[start:end], dtype=DTYPE, device=DEVICE).view(-1, 1)
        y_t = torch.tensor(y_flat[start:end], dtype=DTYPE, device=DEVICE).view(-1, 1)
        z_t = torch.tensor(z_flat[start:end], dtype=DTYPE, device=DEVICE).view(-1, 1)
        t_t = torch.full_like(x_t, float(t_value))

        with torch.no_grad():
            u = model(torch.cat([x_t, y_t, z_t, t_t], dim=1))
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


def draw_isosurface(ax, u, level=0.0, cmap="Reds", title=""):
    n = u.shape[0]
    if not (np.min(u) <= level <= np.max(u)):
        ax.text2D(0.18, 0.5, f"level {level:g} not in range", transform=ax.transAxes)
    else:
        verts, faces, _, values = measure.marching_cubes(
            u, level=level, spacing=(1.0 / n, 1.0 / n, 1.0 / n)
        )
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            faces,
            verts[:, 2],
            cmap=cmap,
            linewidth=0.05,
            antialiased=True,
            shade=True,
        )

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("X", fontsize=9)
    ax.set_ylabel("Y", fontsize=9)
    ax.set_zlabel("Z", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.view_init(elev=25, azim=45)


def plot_pinn_only(plot_times, n=64, level=0.0):
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    y = np.linspace(0.0, 1.0, n, endpoint=False)
    z = np.linspace(0.0, 1.0, n, endpoint=False)
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, len(plot_times) + 1, width_ratios=[1] * len(plot_times) + [0.05])

    for i, t_plot in enumerate(plot_times):
        model = load_model_3d(get_step_id(t_plot))
        u = evaluate_pinn(model, x_grid, y_grid, z_grid, t_plot)
        ax = fig.add_subplot(gs[0, i], projection="3d")
        draw_isosurface(ax, u, level=level, title=f"PINN, t={t_plot:g}")

    mappable = plt.cm.ScalarMappable(cmap=plt.cm.Reds)
    cbar_ax = fig.add_subplot(gs[0, len(plot_times)])
    fig.colorbar(mappable, cax=cbar_ax, label="Isosurface level")
    return fig, "plot_3D.png"


def plot_comparison(plot_times, ref, level=0.0):
    x = ref["x"]
    y = ref["y"]
    z = ref["z"]
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z, indexing="ij")

    fig = plt.figure(figsize=(21, 13))
    gs = GridSpec(
        3,
        len(plot_times) + 1,
        figure=fig,
        width_ratios=[1] * len(plot_times) + [0.05],
    )
    global_max_error = 0.0
    plot_data = []

    for t_plot in plot_times:
        t_eval, u_ref = nearest_reference(ref, t_plot)
        model = load_model_3d(get_step_id(t_eval))
        u_pinn = evaluate_pinn(model, x_grid, y_grid, z_grid, t_eval)
        abs_error = np.abs(u_pinn - u_ref)
        rel_l2 = np.linalg.norm(u_pinn - u_ref) / np.linalg.norm(u_ref)
        max_error = float(np.max(abs_error))
        global_max_error = max(global_max_error, max_error)
        plot_data.append((t_eval, u_ref, u_pinn, abs_error, rel_l2, max_error))

    for col, (t_eval, u_ref, u_pinn, abs_error, rel_l2, max_error) in enumerate(plot_data):
        error_level = 0.5 * max_error

        ax_ref = fig.add_subplot(gs[0, col], projection="3d")
        draw_isosurface(ax_ref, u_ref, level=level, title=f"Reference, t={t_eval:g}")

        ax_pinn = fig.add_subplot(gs[1, col], projection="3d")
        draw_isosurface(
            ax_pinn,
            u_pinn,
            level=level,
            title=f"PINN, rel L2={rel_l2:.2e}",
        )

        ax_error = fig.add_subplot(gs[2, col], projection="3d")
        draw_isosurface(
            ax_error,
            abs_error,
            level=error_level,
            cmap="viridis",
            title=f"|error|, max={max_error:.2e}",
        )

    solution_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=level, vmax=level + 1.0),
        cmap=plt.cm.Reds,
    )
    solution_mappable.set_array([])
    solution_cax = fig.add_subplot(gs[:2, len(plot_times)])
    solution_cbar = fig.colorbar(solution_mappable, cax=solution_cax)
    solution_cbar.set_ticks([level])
    solution_cbar.set_ticklabels([f"{level:g}"])
    solution_cbar.set_label(r"$u=0$ isosurface")

    error_mappable = plt.cm.ScalarMappable(
        norm=Normalize(vmin=0.0, vmax=global_max_error),
        cmap=plt.cm.viridis,
    )
    error_mappable.set_array([])
    error_cax = fig.add_subplot(gs[2, len(plot_times)])
    error_cbar = fig.colorbar(error_mappable, cax=error_cax)
    error_cbar.set_label(r"$|u_{PINN}-u_{ref}|$")

    fig.tight_layout()
    return fig, "plot_3D_compare.png"


if __name__ == "__main__":
    plot_times = [0.0, 0.2, 0.5, 1.0]
    level = 0.0

    ref = load_reference()
    if ref is None:
        fig, save_name = plot_pinn_only(plot_times, n=64, level=level)
        print(
            "\nreference_3d.npz not found; generated PINN-only plot. "
            "Run generate_reference.py first for comparison plots."
        )
    else:
        fig, save_name = plot_comparison(plot_times, ref, level=level)

    save_path = os.path.join(SCRIPT_DIR, save_name)
    fig.savefig(save_path, dpi=350, bbox_inches="tight")
    print(f"\nfigure saved to: {save_path}")
    plt.show()
