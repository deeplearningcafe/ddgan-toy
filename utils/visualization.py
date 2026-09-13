import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_ground_truth_forward_process(
    data: torch.Tensor,
    fwd_coeffs,
    num_timesteps: int = 4,
    num_samples: int = 10000,
    save_path: str = "forward_diffusion_evolution.png",
):
    """Visualizes forward signal degradation across discrete diffusion steps."""
    n_pts = min(num_samples, data.shape[0])
    x_0 = data[:n_pts].detach().clone()
    eps = torch.randn_like(x_0)

    alphas, sigmas, x_t_states = [], [], []
    for step in range(num_timesteps + 1):
        if step == 0:
            a_scalar, s_scalar = 1.0, 0.0
            x_t = x_0
        else:
            a_scalar = float(fwd_coeffs.a_s_cum[step].cpu())
            s_scalar = float(fwd_coeffs.sigmas_cum[step].cpu())
            x_t = a_scalar * x_0 + s_scalar * eps

        alphas.append(a_scalar)
        sigmas.append(s_scalar)
        x_t_states.append(x_t.cpu().numpy())

    n_cols = num_timesteps + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5))

    for idx in range(n_cols):
        ax = axes[idx]
        pts = x_t_states[idx]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=2,
            alpha=0.3,
            color="#1f77b4",
            edgecolors="none",
        )
        ax.set_title(
            f"Step {idx}\nsqrt(alpha)={alphas[idx]:.2f}, sig={sigmas[idx]:.2f}",
            fontsize=9,
        )
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_aspect("equal", "box")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_scatter_kde(samples_2d: np.ndarray, step: int, save_path: str):
    """Dual-panel plot displaying scatter distribution and KDE contour."""
    var_x = float(np.var(samples_2d[:, 0]))
    var_y = float(np.var(samples_2d[:, 1]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax in axes:
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal", "box")

    axes[0].scatter(samples_2d[:, 0], samples_2d[:, 1], s=2, alpha=0.3, color="#1f77b4")
    axes[0].set_title(f"Step {step}: Scatter Points")

    if var_x > 1e-4 and var_y > 1e-4:
        sns.kdeplot(
            x=samples_2d[:, 0],
            y=samples_2d[:, 1],
            cmap="Blues",
            fill=True,
            thresh=0.05,
            ax=axes[1],
            warn_singular=False,
        )
    else:
        axes[1].text(0, 0, "Variance too low for KDE", ha="center", va="center")
    axes[1].set_title(f"Step {step}: KDE Density")

    plt.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_trajectories_and_samples(
    real_data_2d: np.ndarray,
    gen_data_2d: np.ndarray,
    real_traj: list,
    gen_traj: list,
    num_timesteps: int,
    save_filepath: str,
    proj_mat=None,
    max_points: int = 3000,
):
    """Visualizes ground truth vs. generated states along the diffusion path."""
    fig = plt.figure(figsize=(4 * num_timesteps, 8))
    gs = plt.GridSpec(2, num_timesteps, figure=fig, hspace=0.35, wspace=0.25)

    ax_real = fig.add_subplot(gs[0, : num_timesteps // 2])
    ax_gen = fig.add_subplot(gs[0, num_timesteps // 2 :])

    n_pts = min(max_points, real_data_2d.shape[0], gen_data_2d.shape[0])
    ax_real.scatter(
        real_data_2d[:n_pts, 0],
        real_data_2d[:n_pts, 1],
        s=4,
        alpha=0.4,
        color="#1f77b4",
    )
    ax_real.set_title("Ground Truth Samples (t=0)", fontsize=11)
    ax_real.set_xlim(-3.5, 3.5)
    ax_real.set_ylim(-3.5, 3.5)
    ax_real.set_aspect("equal", "box")

    ax_gen.scatter(
        gen_data_2d[:n_pts, 0],
        gen_data_2d[:n_pts, 1],
        s=4,
        alpha=0.4,
        color="#e77148",
    )
    ax_gen.set_title("DDGAN Samples (EMA, t=0)", fontsize=11)
    ax_gen.set_xlim(-3.5, 3.5)
    ax_gen.set_ylim(-3.5, 3.5)
    ax_gen.set_aspect("equal", "box")

    def project(state_tensor):
        if proj_mat is not None:
            return (state_tensor @ proj_mat).cpu().numpy()
        return state_tensor.cpu().numpy()

    for col_idx, step in enumerate(reversed(range(1, num_timesteps + 1))):
        ax_t = fig.add_subplot(gs[1, col_idx])
        r_state = project(real_traj[step])[:n_pts]
        g_state = project(gen_traj[num_timesteps - step])[:n_pts]

        ax_t.scatter(
            r_state[:, 0],
            r_state[:, 1],
            s=3,
            alpha=0.3,
            color="#1f77b4",
            label="Real",
        )
        ax_t.scatter(
            g_state[:, 0],
            g_state[:, 1],
            s=3,
            alpha=0.3,
            color="#e77148",
            label="Generated",
        )
        ax_t.set_title(f"Step t={step} (2D Points)")
        ax_t.set_xlim(-3.5, 3.5)
        ax_t.set_ylim(-3.5, 3.5)
        ax_t.set_aspect("equal", "box")

        if col_idx == 0:
            ax_t.legend(loc="upper left", markerscale=2)

    plt.tight_layout()
    fig.savefig(save_filepath, dpi=180)
    plt.close(fig)
