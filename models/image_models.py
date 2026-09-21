import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbeddings(nn.Module):
    """Sinusoidal embeddings and projection matching toy-diffusion."""

    def __init__(
        self,
        sinusoidal_dim: int,
        output_dim: int,
        max_period: int = 10000,
    ):
        super().__init__()
        self.sinusoidal_dim = sinusoidal_dim
        self.output_dim = output_dim
        half_dim = sinusoidal_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32
        )
        exponent = exponent / half_dim
        self.register_buffer("inv_freq", torch.exp(exponent), persistent=False)
        self.linear_1 = nn.Linear(sinusoidal_dim, output_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(output_dim, output_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim > 1:
            timesteps = timesteps.squeeze()
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        args = timesteps.unsqueeze(1).float() * self.inv_freq.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.linear_2(self.act(self.linear_1(emb)))


class ResnetBlock2D(nn.Module):
    """Residual block with conditioning projection from toy-diffusion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embeddings_channels: int,
        norm_num_groups: int = 8,
        eps: float = 1e-6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(norm_num_groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_emb_proj = nn.Linear(time_embeddings_channels, out_channels)
        self.nonlinearity = nn.SiLU()
        self.norm2 = nn.GroupNorm(norm_num_groups, out_channels, eps=eps)
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor = None,
    ) -> torch.Tensor:
        h = self.conv1(self.nonlinearity(self.norm1(x)))
        if temb is not None:
            h = h + self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(self.nonlinearity(self.norm2(h))))
        res = x if self.conv_shortcut is None else self.conv_shortcut(x)
        return h + res


class MinibatchStdDev(nn.Module):
    """Computes batch variance statistics to penalize mode collapse."""

    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        group = min(b, self.group_size)
        if group <= 1:
            return torch.cat(
                [x, torch.zeros((b, 1, h, w), device=x.device, dtype=x.dtype)],
                dim=1,
            )
        y = x.view(group, -1, 1, c, h, w)
        y = torch.var(y, dim=0, unbiased=False)
        y = torch.sqrt(y + 1e-8)
        y = y.mean(dim=[2, 3, 4], keepdim=True).squeeze(2)
        y = y.repeat(group, 1, h, w)
        return torch.cat([x, y], dim=1)


class GeneratorUNet2D(nn.Module):
    """Multimodal conditional U-Net adhering to toy-diffusion notation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        block_out_channels: list[int] = [64, 128],
        time_embeddings_channels: int = 128,
        z_dim: int = 100,
        norm_num_groups: int = 8,
        norm_eps: float = 1e-5,
        vanilla_gan: bool = False,
    ):
        super().__init__()
        self.vanilla_gan = vanilla_gan
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)
        self.z_embedding = nn.Sequential(
            nn.Linear(z_dim, time_embeddings_channels),
            nn.SiLU(),
            nn.Linear(time_embeddings_channels, time_embeddings_channels),
        )
        if not vanilla_gan:
            self.time_embedding = TimeEmbeddings(
                sinusoidal_dim=block_out_channels[0],
                output_dim=time_embeddings_channels,
            )

        # Down blocks
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[0],
                                    block_out_channels[0],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                        "downsamplers": nn.ModuleList(
                            [
                                nn.Conv2d(
                                    block_out_channels[0],
                                    block_out_channels[0],
                                    3,
                                    stride=2,
                                    padding=1,
                                )
                            ]
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[0],
                                    block_out_channels[1],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                        "downsamplers": nn.ModuleList(
                            [
                                nn.Conv2d(
                                    block_out_channels[1],
                                    block_out_channels[1],
                                    3,
                                    stride=2,
                                    padding=1,
                                )
                            ]
                        ),
                    }
                ),
            ]
        )

        # Mid block
        self.mid_block = nn.ModuleDict(
            {
                "resnets": nn.ModuleList(
                    [
                        ResnetBlock2D(
                            block_out_channels[1],
                            block_out_channels[1],
                            time_embeddings_channels,
                            norm_num_groups=norm_num_groups,
                            eps=norm_eps,
                        )
                    ]
                )
            }
        )

        # Up blocks
        self.up_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "upsamplers": nn.ModuleList(
                            [nn.Upsample(scale_factor=2.0, mode="nearest")]
                        ),
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[1] * 2,
                                    block_out_channels[1],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "upsamplers": nn.ModuleList(
                            [nn.Upsample(scale_factor=2.0, mode="nearest")]
                        ),
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[1] + block_out_channels[0],
                                    block_out_channels[0],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                    }
                ),
            ]
        )

        self.conv_norm_out = nn.GroupNorm(
            norm_num_groups, block_out_channels[0], eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

    def forward(
        self,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        z_emb = self.z_embedding(z)
        if self.vanilla_gan:
            t_emb = z_emb
            x_in = torch.zeros(
                (z.shape[0], self.in_channels, 32, 32),
                device=z.device,
                dtype=z.dtype,
            )
        else:
            t_emb = self.time_embedding(t) + z_emb
            x_in = x_tp1

        x = self.conv_in(x_in)
        skips = [x]

        for block in self.down_blocks:
            for resnet in block["resnets"]:
                x = resnet(x, t_emb)
                skips.append(x)
            for downsampler in block["downsamplers"]:
                x = downsampler(x)

        for resnet in self.mid_block["resnets"]:
            x = resnet(x, t_emb)

        for block in self.up_blocks:
            for upsampler in block["upsamplers"]:
                x = upsampler(x)
            res_skip = skips.pop()
            x = torch.cat([x, res_skip], dim=1)
            for resnet in block["resnets"]:
                x = resnet(x, t_emb)

        out = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        return torch.tanh(out)


class Discriminator2D(nn.Module):
    """Classifier discriminator using toy-diffusion layers and notation."""

    def __init__(
        self,
        in_channels: int = 1,
        block_out_channels: list[int] = [64, 128],
        time_embeddings_channels: int = 128,
        norm_num_groups: int = 8,
        norm_eps: float = 1e-5,
        vanilla_gan: bool = False,
    ):
        super().__init__()
        self.vanilla_gan = vanilla_gan
        input_channels = in_channels if vanilla_gan else in_channels * 2

        if not vanilla_gan:
            self.time_embedding = TimeEmbeddings(
                sinusoidal_dim=block_out_channels[0],
                output_dim=time_embeddings_channels,
            )

        self.conv_in = nn.Conv2d(input_channels, block_out_channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[0],
                                    block_out_channels[0],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                        "downsamplers": nn.ModuleList(
                            [
                                nn.Conv2d(
                                    block_out_channels[0],
                                    block_out_channels[1],
                                    3,
                                    stride=2,
                                    padding=1,
                                )
                            ]
                        ),
                    }
                ),
                nn.ModuleDict(
                    {
                        "resnets": nn.ModuleList(
                            [
                                ResnetBlock2D(
                                    block_out_channels[1],
                                    block_out_channels[1],
                                    time_embeddings_channels,
                                    norm_num_groups=norm_num_groups,
                                    eps=norm_eps,
                                )
                            ]
                        ),
                        "downsamplers": nn.ModuleList(
                            [
                                nn.Conv2d(
                                    block_out_channels[1],
                                    block_out_channels[1],
                                    3,
                                    stride=2,
                                    padding=1,
                                )
                            ]
                        ),
                    }
                ),
            ]
        )

        self.mbstd = MinibatchStdDev()
        self.conv_norm_out = nn.GroupNorm(
            norm_num_groups, block_out_channels[1] + 1, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.linear_out = nn.Linear(block_out_channels[1] + 1, 1)

    def forward(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor = None,
        t: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.vanilla_gan:
            x_in = x_t
            t_emb = None
        else:
            x_in = torch.cat([x_t, x_tp1], dim=1)
            t_emb = self.time_embedding(t)

        x = self.conv_in(x_in)
        for block in self.down_blocks:
            for resnet in block["resnets"]:
                x = resnet(x, t_emb)
            for downsampler in block["downsamplers"]:
                x = downsampler(x)

        x = self.mbstd(x)
        x = self.conv_act(self.conv_norm_out(x))
        pooled = x.sum(dim=[2, 3])
        return self.linear_out(pooled).squeeze(-1)
