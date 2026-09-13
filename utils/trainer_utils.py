import copy
import torch
import torch.optim as optim
from ddgan_toy.models.mlp import GeneratorND, DiscriminatorND
from ddgan_toy.paths.scheduler import (
    DDPMSchedule,
    LinearSchedule,
    CosineSchedule,
)


class EMA:
    """Tracks Exponential Moving Average of generator network weights."""

    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)


def gpu_setup(
    device: str = "cuda",
    use_amp: bool = False,
) -> tuple[torch.dtype, bool]:
    """Validates GPU capability and configures Tensor Cores & AMP."""
    autocast_dtype = torch.float32
    enabled = False

    if device == "cuda" and torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        # only support bf16 of fp32
        if cap[0] >= 8:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if use_amp and torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
                enabled = True

    return autocast_dtype, enabled


def build_models(
    data_dim: int,
    z_dim: int,
    hidden_dim: int,
    vanilla_gan: bool,
    device: torch.device,
):
    """Instantiates G and D architectures."""
    net_g = GeneratorND(
        data_dim=data_dim,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        vanilla_gan=vanilla_gan,
    ).to(device)
    net_d = DiscriminatorND(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        vanilla_gan=vanilla_gan,
    ).to(device)
    return net_g, net_d


def build_scheduler(
    scheduler_type: str,
    num_timesteps: int,
    device: torch.device,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
):
    """Factory creating discrete diffusion schedule."""
    stype = scheduler_type.lower()
    if stype == "ddpm":
        return DDPMSchedule(num_timesteps, device, beta_min=beta_min, beta_max=beta_max)
    elif stype in ["linear", "fm", "flow_matching"]:
        return LinearSchedule(num_timesteps, device)
    elif stype == "cosine":
        return CosineSchedule(num_timesteps, device)
    raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_optimizers(net_g, net_d, lr_g: float, lr_d: float):
    """Creates Adam optimizers configured with TTUR."""
    opt_g = optim.Adam(net_g.parameters(), lr=lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(net_d.parameters(), lr=lr_d, betas=(0.5, 0.999))
    return opt_g, opt_d


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Computes total L2 gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.detach().norm(2).item() ** 2
    return total_norm**0.5
