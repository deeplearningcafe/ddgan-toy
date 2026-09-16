import torch
import numpy as np
from abc import ABC, abstractmethod


class DiffusionCoefficients:
    """Forward diffusion schedule coefficients."""

    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.a_s = torch.sqrt(self.alphas)
        self.sigmas = torch.sqrt(betas)
        self.a_s_cum = torch.cumprod(self.a_s, dim=0)
        self.sigmas_cum = torch.sqrt(torch.clamp(1.0 - self.a_s_cum**2, min=0.0))


class PosteriorCoefficients:
    """Closed-form reverse posterior coefficients q(x_t | x_{t+1}, x_0)."""

    def __init__(self, betas: torch.Tensor, device: torch.device):
        self.betas = betas[1:]
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]],
            dim=0,
        )
        self.var = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.var[0] = 0.0

        self.coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )


class BaseSchedule(ABC):
    """Abstract schedule class generating forward/posterior constants."""

    def __init__(self, num_timesteps: int, device: torch.device):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = self._compute_betas().to(device)
        self.fwd = DiffusionCoefficients(self.betas)
        self.pos = PosteriorCoefficients(self.betas, device)

    @abstractmethod
    def _compute_betas(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_scheduler_type(self) -> str:
        pass


class DDPMSchedule(BaseSchedule):
    """Variance Preserving VP-SDE schedule (Xiao et al., 2022)."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        eps = 1e-3
        t = np.arange(0, self.num_timesteps + 1, dtype=np.float64)
        t = (t / self.num_timesteps) * (1.0 - eps) + eps
        log_mean = (
            -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        )
        alpha_bars = torch.exp(2.0 * torch.from_numpy(log_mean))
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor([1e-8], dtype=torch.float64)
        return torch.cat([first, betas]).float()

    def get_scheduler_type(self) -> str:
        return f"ddpm_vp(bmin={self.beta_min},bmax={self.beta_max})"


class LinearSchedule(BaseSchedule):
    """Flow-matching inspired linear variance decay schedule."""

    def _compute_betas(self) -> torch.Tensor:
        eps_final = 1e-4
        t = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        alpha_bars = torch.from_numpy(1.0 - t * (1.0 - eps_final)).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return "linear_uniform"


class CosineSchedule(BaseSchedule):
    """Cosine schedule for smoother boundary degradation."""

    def __init__(self, num_timesteps: int, device: torch.device, s: float = 0.008):
        self.s = s
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.arange(self.num_timesteps + 1, dtype=np.float64)
        f = (
            np.cos(
                ((steps / self.num_timesteps) + self.s) / (1.0 + self.s) * np.pi / 2.0
            )
            ** 2
        )
        alpha_bars = torch.from_numpy(f / f[0]).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return f"cosine(s={self.s})"


class GVPSchedule(BaseSchedule):
    """Generalized Variance Preserving (GVP) trigonometric schedule (SiT)."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        eps_final: float = 1e-4,
    ):
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        # alpha_t = cos(pi / 2 * t), sigma_t = sin(pi / 2 * t)
        # alpha_bars = alpha_t^2 = cos^2(pi / 2 * t)
        alpha_bars = np.cos(0.5 * np.pi * steps) ** 2
        alpha_bars = np.clip(alpha_bars, self.eps_final, 1.0)
        alpha_bars = torch.from_numpy(alpha_bars).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return "gvp_sit"


class PowerSchedule(BaseSchedule):
    """Polynomial schedule prioritizing mode separation at low noise."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        power: float = 2.0,
        eps_final: float = 1e-4,
    ):
        self.power = power
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        alpha_bars = torch.from_numpy(
            1.0 - (1.0 - self.eps_final) * (steps**self.power)
        ).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return f"power(p={self.power})"


class SigmoidSchedule(BaseSchedule):
    """Sigmoid schedule providing smooth S-curve noise transitions."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        start: float = -3.0,
        end: float = 3.0,
        tau: float = 1.0,
        eps_final: float = 1e-4,
    ):
        self.start = start
        self.end = end
        self.tau = tau
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        v = self.start + steps * (self.end - self.start)
        sig = 1.0 / (1.0 + np.exp(v / self.tau))
        norm_sig = (sig - sig[-1]) / (sig[0] - sig[-1])
        alpha_bars = torch.from_numpy(
            norm_sig * (1.0 - self.eps_final) + self.eps_final
        ).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return f"sigmoid(start={self.start},end={self.end})"


class KarrasVPSchedule(BaseSchedule):
    """Karras EDM polynomial noise schedule adapted to VP diffusion."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        sigma_min: float = 0.02,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        eps_final: float = 1e-4,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        inv_rho = 1.0 / self.rho
        sigmas = (
            self.sigma_min**inv_rho
            + steps * (self.sigma_max**inv_rho - self.sigma_min**inv_rho)
        ) ** self.rho
        raw_alpha = 1.0 / (1.0 + sigmas**2)
        norm_alpha = (raw_alpha - raw_alpha[-1]) / (raw_alpha[0] - raw_alpha[-1])
        alpha_bars = torch.from_numpy(
            norm_alpha * (1.0 - self.eps_final) + self.eps_final
        ).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return f"karras_vp(rho={self.rho})"


class GeometricSchedule(BaseSchedule):
    """Geometric/exponential noise progression adapted to VP diffusion."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        sigma_min: float = 0.05,
        sigma_max: float = 20.0,
        eps_final: float = 1e-4,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        sigmas = self.sigma_min * ((self.sigma_max / self.sigma_min) ** steps)
        raw_alpha = 1.0 / (1.0 + sigmas**2)
        norm_alpha = (raw_alpha - raw_alpha[-1]) / (raw_alpha[0] - raw_alpha[-1])
        alpha_bars = torch.from_numpy(
            norm_alpha * (1.0 - self.eps_final) + self.eps_final
        ).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return f"geometric(min={self.sigma_min},max={self.sigma_max})"


class LinearSNRSchedule(BaseSchedule):
    """Linear amplitude schedule: sqrt(alpha_bar) decays linearly."""

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device,
        eps_final: float = 1e-4,
    ):
        self.eps_final = eps_final
        super().__init__(num_timesteps, device)

    def _compute_betas(self) -> torch.Tensor:
        steps = np.linspace(0.0, 1.0, self.num_timesteps + 1, dtype=np.float64)
        sqrt_alpha = 1.0 - steps * (1.0 - np.sqrt(self.eps_final))
        alpha_bars = torch.from_numpy(sqrt_alpha**2).float()
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        betas = torch.clamp(betas, min=1e-5, max=0.999)
        first = torch.tensor([1e-8], dtype=torch.float32)
        return torch.cat([first, betas])

    def get_scheduler_type(self) -> str:
        return "linear_snr"
