import numpy as np
import torch
from scipy.spatial.distance import cdist


def compute_precision_recall(
    generated_samples, gt_samples, k: int = 3
) -> tuple[float, float]:
    """Computes k-NN manifold precision and recall."""
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()
    if isinstance(gt_samples, torch.Tensor):
        gt_samples = gt_samples.detach().cpu().numpy()

    gt_dist = cdist(gt_samples, gt_samples)
    np.fill_diagonal(gt_dist, np.inf)
    gt_radii = np.partition(gt_dist, k - 1, axis=1)[:, k - 1]

    gen_dist = cdist(generated_samples, generated_samples)
    np.fill_diagonal(gen_dist, np.inf)
    gen_radii = np.partition(gen_dist, k - 1, axis=1)[:, k - 1]

    d_gen_to_gt = cdist(generated_samples, gt_samples)
    in_gt_manifold = np.any(d_gen_to_gt <= gt_radii.reshape(1, -1), axis=1)
    precision = float(np.mean(in_gt_manifold))

    d_gt_to_gen = d_gen_to_gt.T
    in_gen_manifold = np.any(d_gt_to_gen <= gen_radii.reshape(1, -1), axis=1)
    recall = float(np.mean(in_gen_manifold))
    return precision, recall


def chamfer_distance(set1, set2, is_trajectory: bool = False) -> float:
    """Computes symmetric squared Chamfer Distance."""
    if isinstance(set1, torch.Tensor):
        set1 = set1.detach().cpu().numpy()
    if isinstance(set2, torch.Tensor):
        set2 = set2.detach().cpu().numpy()

    if is_trajectory:
        if set1.ndim == 3:
            set1 = set1.reshape(set1.shape[0], -1)
        if set2.ndim == 3:
            set2 = set2.reshape(set2.shape[0], -1)

    dist_matrix = cdist(set1, set2, metric="euclidean")
    min1 = np.min(dist_matrix, axis=1)
    min2 = np.min(dist_matrix, axis=0)
    return float(np.mean(min1**2) + np.mean(min2**2))


def compute_curvature(trajectories) -> float:
    """Computes trajectory straightness ratio (1.0 = perfectly straight)."""
    if isinstance(trajectories, np.ndarray):
        trajectories = torch.from_numpy(trajectories)
    if trajectories.ndim > 3:
        b, t = trajectories.shape[:2]
        trajectories = trajectories.view(b, t, -1)

    start = trajectories[:, 0, :]
    end = trajectories[:, -1, :]
    displacement = torch.norm(end - start, dim=-1)

    diffs = trajectories[:, 1:, :] - trajectories[:, :-1, :]
    path_len = torch.norm(diffs, dim=-1).sum(dim=-1)
    return float((path_len / (displacement + 1e-8)).mean().item())
