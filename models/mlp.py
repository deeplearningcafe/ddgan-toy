import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional time step embedding."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * -(math.log(10000) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class GeneratorND(nn.Module):
    """Conditional generator for DDGAN supporting arbitrary dimensions."""

    def __init__(
        self,
        data_dim: int = 2,
        z_dim: int = 2,
        hidden_dim: int = 512,
        t_dim: int = 64,
        vanilla_gan: bool = False,
    ):
        super().__init__()
        self.vanilla_gan = vanilla_gan
        in_dim = z_dim if vanilla_gan else (data_dim + z_dim + hidden_dim)

        if not vanilla_gan:
            self.time_embed = SinusoidalTimeEmbedding(t_dim)
            self.t_mlp = nn.Sequential(
                nn.Linear(t_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(
        self,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        if self.vanilla_gan:
            return self.net(z)

        t_emb = self.t_mlp(self.time_embed(t))
        return self.net(torch.cat([x_tp1, z, t_emb], dim=-1))


class DiscriminatorND(nn.Module):
    """Conditional discriminator for DDGAN."""

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 512,
        t_dim: int = 64,
        vanilla_gan: bool = False,
    ):
        super().__init__()
        self.vanilla_gan = vanilla_gan
        in_dim = data_dim if vanilla_gan else (data_dim * 2 + hidden_dim)

        if not vanilla_gan:
            self.time_embed = SinusoidalTimeEmbedding(t_dim)
            self.t_mlp = nn.Sequential(
                nn.Linear(t_dim, hidden_dim),
                nn.LeakyReLU(0.2),
            )

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.vanilla_gan:
            return self.net(x_t).squeeze(-1)

        t_emb = self.t_mlp(self.time_embed(t))
        return self.net(torch.cat([x_t, x_tp1, t_emb], dim=-1)).squeeze(-1)
