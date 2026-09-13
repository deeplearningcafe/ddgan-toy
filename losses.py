import torch
import torch.nn as nn
import torch.nn.functional as F
from paths.sampling import (
    q_sample_pairs,
    sample_posterior,
    forward_diffuse_step,
    convert_pred_to_x0,
)


class DDGANLoss(nn.Module):
    """Handles Discriminator (R1 penalty) and Generator (AFD cycle) losses."""

    def __init__(
        self,
        fwd_coeffs,
        pos_coeffs,
        num_timesteps: int = 4,
        r1_gamma: float = 0.05,
        afd_weight: float = 0.5,
        pred_target: str = "x0",
        vanilla_gan: bool = False,
    ):
        super().__init__()
        self.fwd_coeffs = fwd_coeffs
        self.pos_coeffs = pos_coeffs
        self.num_timesteps = num_timesteps
        self.r1_gamma = r1_gamma
        self.afd_weight = afd_weight
        self.pred_target = pred_target
        self.vanilla_gan = vanilla_gan

    def discriminator_loss(
        self,
        net_d,
        net_g,
        x_0: torch.Tensor,
        z_dim: int,
    ):
        batch_size = x_0.shape[0]
        device = x_0.device

        if self.vanilla_gan:
            x_real = x_0.requires_grad_(True)
            d_real = net_d(x_real)
            loss_d_real = F.softplus(-d_real).mean()

            grad_real = torch.autograd.grad(
                outputs=d_real.sum(), inputs=x_real, create_graph=True
            )[0]
            r1_pen = grad_real.pow(2).sum(dim=1).mean()

            with torch.no_grad():
                z = torch.randn(batch_size, z_dim, device=device)
                x_fake = net_g(None, None, z)

            d_fake = net_d(x_fake)
            loss_d_fake = F.softplus(d_fake).mean()
            x_tp1 = None
            t = None
        else:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            x_t, x_tp1 = q_sample_pairs(self.fwd_coeffs, x_0, t)
            x_t.requires_grad_(True)

            d_real = net_d(x_t, x_tp1.detach(), t)
            loss_d_real = F.softplus(-d_real).mean()

            grad_real = torch.autograd.grad(
                outputs=d_real.sum(), inputs=x_t, create_graph=True
            )[0]
            r1_pen = grad_real.pow(2).sum(dim=1).mean()

            with torch.no_grad():
                z = torch.randn(batch_size, z_dim, device=device)
                raw_pred = net_g(x_tp1.detach(), t, z)
                x_0_pred = convert_pred_to_x0(
                    raw_pred,
                    x_tp1.detach(),
                    self.fwd_coeffs,
                    t,
                    self.pred_target,
                )
                x_t_fake = sample_posterior(
                    self.pos_coeffs, x_0_pred, x_tp1.detach(), t
                )

            d_fake = net_d(x_t_fake.detach(), x_tp1.detach(), t)
            loss_d_fake = F.softplus(d_fake).mean()

        total_loss = loss_d_real + loss_d_fake + 0.5 * self.r1_gamma * r1_pen
        diagnostics = {
            "loss_d": total_loss,
            "loss_d_real": loss_d_real,
            "loss_d_fake": loss_d_fake,
            "r1_pen": r1_pen,
            "d_real_mean": d_real.mean(),
            "d_fake_mean": d_fake.mean(),
            "x_tp1": x_tp1,
            "t": t,
        }
        return total_loss, diagnostics

    def generator_loss(
        self,
        net_d,
        net_g,
        x_tp1: torch.Tensor,
        t: torch.Tensor,
        z_dim: int,
        batch_size: int,
        device: torch.device,
    ):
        if self.vanilla_gan:
            z_g = torch.randn(batch_size, z_dim, device=device)
            x_fake_g = net_g(None, None, z_g)
            d_fake_g = net_d(x_fake_g)
            loss_g = F.softplus(-d_fake_g).mean()
            loss_afd = torch.tensor(0.0, device=device)
            loss_adv = loss_g
        else:
            z_g = torch.randn(batch_size, z_dim, device=device)
            raw_pred_g = net_g(x_tp1.detach(), t, z_g)
            x_0_pred_g = convert_pred_to_x0(
                raw_pred_g,
                x_tp1.detach(),
                self.fwd_coeffs,
                t,
                self.pred_target,
            )
            x_t_fake_g = sample_posterior(
                self.pos_coeffs, x_0_pred_g, x_tp1.detach(), t
            )

            d_fake_g = net_d(x_t_fake_g, x_tp1.detach(), t)
            loss_adv = F.softplus(-d_fake_g).mean()

            x_tp1_rec = forward_diffuse_step(self.fwd_coeffs, x_t_fake_g, t)
            loss_afd = F.mse_loss(x_tp1_rec, x_tp1.detach())
            loss_g = loss_adv + self.afd_weight * loss_afd

        diagnostics = {
            "loss_g": loss_g,
            "loss_adv": loss_adv,
            "loss_afd": loss_afd,
        }
        return loss_g, diagnostics
