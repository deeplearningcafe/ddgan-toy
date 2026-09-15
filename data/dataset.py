import numpy as np
import torch
from torch.utils.data import Dataset


class Synthetic2DDataset(Dataset):
    """Generates synthetic 2D point distributions with optional projection."""

    def __init__(
        self,
        name: str = "grid",
        n_samples: int = 250000,
        projection_dim: int = 0,
    ):
        super().__init__()
        self.name = name.lower()
        self.projection_dim = projection_dim
        self.data, self.proj_mat = self._generate_data(n_samples)

    def _generate_data(self, n_samples: int):
        if self.name in ["grid", "25gaussians"]:
            grid_vals = np.linspace(-2.0, 2.0, 5)
            grid_x, grid_y = np.meshgrid(grid_vals, grid_vals)
            centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
            idx = np.random.choice(len(centers), n_samples)
            noise = np.random.randn(n_samples, 2) * 0.05
            data = centers[idx] + noise

        elif self.name == "circle":
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            centers = np.stack([np.cos(angles) * 2.0, np.sin(angles) * 2.0], 1)
            idx = np.random.choice(8, n_samples)
            data = centers[idx] + np.random.randn(n_samples, 2) * 0.1

        elif self.name == "imbalanced":
            angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
            centers = np.stack([np.cos(angles) * 2.0, np.sin(angles) * 2.0], 1)
            probs = [0.65] + [0.35 / 7.0] * 7
            idx = np.random.choice(8, n_samples, p=probs)
            data = centers[idx] + np.random.randn(n_samples, 2) * 0.1

        elif self.name == "spiral":
            theta = np.sqrt(np.random.rand(n_samples)) * 4.0 * np.pi
            r = 2.0 * theta + np.pi
            x = np.cos(theta) * r
            y = np.sin(theta) * r
            raw = np.stack([x, y], axis=1) + np.random.randn(n_samples, 2) * 0.2
            data = (raw - raw.mean(0)) / raw.std(0) * 1.5

        elif self.name == "gmm_long_tail":
            num_modes = 10
            centers = np.stack(
                [np.arange(num_modes) * 2.0 - (num_modes - 1), np.zeros(num_modes)],
                axis=1,
            )
            probs = np.array([1.0 / (2 ** (i + 1)) for i in range(num_modes)])
            probs /= probs.sum()
            idx = np.random.choice(num_modes, n_samples, p=probs)
            raw = centers[idx] + np.random.randn(n_samples, 2) * 0.2
            data = (raw - raw.mean(0)) / raw.std(0)

        elif self.name == "pinwheel":
            radial_std, tangential_std = 0.3, 0.1
            num_classes = 5
            num_per = n_samples // num_classes
            rate = 0.25
            rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
            feats = np.random.randn(num_classes * num_per, 2) * [
                radial_std,
                tangential_std,
            ]
            feats[:, 0] += 1
            labels = np.repeat(np.arange(num_classes), num_per)
            angles = rads[labels] + rate * np.exp(feats[:, 0])
            rot = np.stack(
                [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)],
                1,
            ).reshape(-1, 2, 2)
            pts = np.einsum("ti,tij->tj", feats, rot)
            data = (pts - pts.mean(0)) / pts.std(0)

        elif self.name in "chessboard":
            # 4x4 [-2, 2] x [-2, 2] (8 active tiles)
            active_tiles = [
                (i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0
            ]
            tile_indices = np.random.choice(len(active_tiles), n_samples)
            chosen_tiles = np.array(
                [active_tiles[idx] for idx in tile_indices],
                dtype=np.float32,
            )
            offsets = np.random.uniform(0.0, 1.0, size=(n_samples, 2)).astype(
                np.float32
            )
            # Center grid to [-2, 2] across both dimensions
            data = (chosen_tiles + offsets) - 2.0
            data = (data - data.mean(0)) / data.std(0)

        else:
            raise ValueError(f"Unknown synthetic dataset: {self.name}")

        data_tensor = torch.from_numpy(data).float()
        proj_mat = None
        if self.projection_dim > 2:
            rand_mat = torch.randn(self.projection_dim, 2)
            proj_mat, _ = torch.linalg.qr(rand_mat, mode="reduced")
            data_tensor = data_tensor @ proj_mat.T

        return data_tensor, proj_mat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
