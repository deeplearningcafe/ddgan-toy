import os
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

from losses import DDGANLoss
from paths.sampling import (
    sample_from_model,
    get_real_forward_trajectories,
)
from evaluation import (
    compute_precision_recall,
    chamfer_distance,
    compute_curvature,
)
from utils.logging_utils import Logger
from utils.trainer_utils import (
    EMA,
    gpu_setup,
    build_models,
    build_scheduler,
    build_optimizers,
    compute_grad_norm,
)
from utils.visualization import (
    plot_ground_truth_forward_process,
    plot_scatter_kde,
    plot_trajectories_and_samples,
)


class DDGANTrainer:
    """Modular trainer coordinating GAN optimization, metrics, and diagnostics."""

    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.save_dir = config.get("save_path", "./results_ddgan")
        Logger.setup_logging(self.save_dir, "run_log.txt")

        self.dtype, self.amp_enabled = gpu_setup(
            device=self.config.get("device", "cuda"),
            use_amp=self.config.get("use_amp", False),
        )

        self.num_timesteps = config.get("num_timesteps", 4)
        self.vanilla_gan = config.get("vanilla_gan", False)
        self.pred_target = config.get("pred_target", "x0")
        self.eval_interval = config.get("eval_interval", 5000)
        self.log_interval = config.get("log_interval", 1000)
        self.afd_weight = config.get("afd_weight", 0.0)
        self.ema_decay = config.get("ema_decay", 0.999)
        self.lr_d = config.get("lr_d", 1e-4)
        self.lr_g = config.get("lr_g", 1e-4)
        self.sched_name = config.get("scheduler_type", "ddpm")

        self.schedule = None
        self.fwd_coeffs = None
        self.pos_coeffs = None
        if not self.vanilla_gan:
            self.schedule = build_scheduler(
                scheduler_type=self.sched_name,
                num_timesteps=self.num_timesteps,
                device=self.device,
                beta_min=config.get("beta_min", 0.1),
                beta_max=config.get("beta_max", 20.0),
            )
            self.fwd_coeffs = self.schedule.fwd
            self.pos_coeffs = self.schedule.pos

        self.proj_dim = config.get("projection_dim", 0)
        self.data_dim = self.proj_dim if self.proj_dim > 2 else 2
        self.z_dim = max(2, self.data_dim // 2) if self.proj_dim > 2 else 2

        self.net_g, self.net_d = build_models(
            data_dim=self.data_dim,
            z_dim=self.z_dim,
            hidden_dim=config.get("hidden_dim", 512),
            vanilla_gan=self.vanilla_gan,
            device=self.device,
        )
        self.ema_g = EMA(self.net_g, decay=self.ema_decay)

        self.opt_g, self.opt_d = build_optimizers(
            self.net_g,
            self.net_d,
            lr_g=self.lr_g,
            lr_d=self.lr_d,
        )
        self.loss_fn = DDGANLoss(
            fwd_coeffs=self.fwd_coeffs,
            pos_coeffs=self.pos_coeffs,
            num_timesteps=self.num_timesteps,
            r1_gamma=config.get("r1_gamma", 0.05),
            afd_weight=self.afd_weight,
            pred_target=self.pred_target,
            vanilla_gan=self.vanilla_gan,
        )

    def train_step(self, x_0: torch.Tensor) -> dict:
        """Executes a single balanced D and G optimization step."""
        x_0 = x_0.to(self.device, non_blocking=True)
        batch_size = x_0.shape[0]

        # 1. Train Discriminator
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp_enabled,
        ):
            loss_d, d_diag = self.loss_fn.discriminator_loss(
                self.net_d, self.net_g, x_0, self.z_dim
            )

        self.opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        self.opt_d.step()
        grad_d = compute_grad_norm(self.net_d)

        # 2. Train Generator
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=self.amp_enabled,
        ):
            loss_g, g_diag = self.loss_fn.generator_loss(
                self.net_d,
                self.net_g,
                d_diag["x_tp1"],
                d_diag["t"],
                self.z_dim,
                batch_size,
                self.device,
            )

        self.opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        self.opt_g.step()
        grad_g = compute_grad_norm(self.net_g)

        self.ema_g.update(self.net_g)

        return {
            "loss_d": loss_d.item(),
            "loss_d_real": d_diag["loss_d_real"].item(),
            "loss_d_fake": d_diag["loss_d_fake"].item(),
            "r1": d_diag["r1_pen"].item(),
            "loss_g": loss_g.item(),
            "loss_adv": g_diag["loss_adv"].item(),
            "loss_afd": g_diag["loss_afd"].item(),
            "d_real": d_diag["d_real_mean"].item(),
            "d_fake": d_diag["d_fake_mean"].item(),
            "grad_d": grad_d,
            "grad_g": grad_g,
        }

    def train(self, dataset, dataloader: DataLoader, iterations: int):
        mode_name = (
            "Vanilla GAN" if self.vanilla_gan else f"DDGAN (T={self.num_timesteps})"
        )
        logging.info(
            f"Training {mode_name} with AFD={self.afd_weight}, {iterations} steps | "
            f"EMA={self.ema_decay}, lr_d={self.lr_d}, lr_g={self.lr_g}"
            f"| Data Dim: {self.data_dim} | "
            f"Projection: {self.proj_dim > 2} | "
            f"Schedule: {self.sched_name} | AMP: {self.amp_enabled} ({self.dtype})"
        )

        proj_mat = (
            dataset.proj_mat.to(self.device) if dataset.proj_mat is not None else None
        )

        if not self.vanilla_gan:
            plot_ground_truth_forward_process(
                data=dataset.data,
                fwd_coeffs=self.fwd_coeffs,
                num_timesteps=self.num_timesteps,
                save_path=os.path.join(
                    self.save_dir, "forward_diffusion_evolution.png"
                ),
            )

        data_iter = iter(dataloader)
        for step in range(1, iterations + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            metrics = self.train_step(batch)

            if step % self.log_interval == 0 or step == 1:
                logging.info(
                    f"Step {step:5d} | D: {metrics['loss_d']:.4f} "
                    f"(R: {metrics['loss_d_real']:.3f}, "
                    f"F: {metrics['loss_d_fake']:.3f}) | "
                    f"G: {metrics['loss_g']:.3f} "
                    f"(Adv: {metrics['loss_adv']:.3f}, "
                    f"AFD: {metrics['loss_afd']:.3f}) | "
                    f"R1: {metrics['r1']:.4f} | "
                    f"D(x): {metrics['d_real']:+.2f} | "
                    f"D(G): {metrics['d_fake']:+.2f} | "
                    f"|gD|: {metrics['grad_d']:.2f} | "
                    f"|gG|: {metrics['grad_g']:.2f}"
                )

            if step % self.eval_interval == 0 or step == iterations:
                self.evaluate(dataset, proj_mat, step)

    def evaluate(self, dataset, proj_mat, step: int):
        eval_samples = 2048
        raw_samples, gen_traj = sample_from_model(
            pos_coeffs=self.pos_coeffs,
            fwd_coeffs=self.fwd_coeffs,
            net_g=self.ema_g.shadow,
            num_timesteps=self.num_timesteps,
            num_samples=eval_samples,
            data_dim=self.data_dim,
            z_dim=self.z_dim,
            device=self.device,
            pred_target=self.pred_target,
            vanilla_gan=self.vanilla_gan,
            return_trajectory=True,
        )

        samples_2d = (
            (raw_samples @ proj_mat).cpu().numpy()
            if proj_mat is not None
            else raw_samples.cpu().numpy()
        )
        gt_high = dataset.data[:eval_samples]
        gt_2d = (
            (gt_high @ dataset.proj_mat).numpy()
            if dataset.proj_mat is not None
            else gt_high.numpy()
        )

        c_dist = chamfer_distance(samples_2d, gt_2d)
        prec, rec = compute_precision_recall(samples_2d, gt_2d, k=5)
        curv = compute_curvature(torch.stack(gen_traj, dim=1))

        logging.info(
            f"--- Evaluation at Step {step} ---\n"
            f"  > Chamfer Dist: {c_dist:.4f}\n"
            f"  > Precision:    {prec:.4f}\n"
            f"  > Recall:       {rec:.4f}\n"
            f"  > Curvature:    {curv:.4f}\n"
            f"--------------------------------"
        )

        plot_scatter_kde(
            samples_2d=samples_2d,
            step=step,
            save_path=os.path.join(self.save_dir, f"step_{step}.png"),
        )

        if not self.vanilla_gan:
            idx = torch.randint(0, len(dataset), (eval_samples,))
            real_batch = dataset.data[idx].to(self.device)
            real_traj = get_real_forward_trajectories(
                self.fwd_coeffs, real_batch, self.num_timesteps
            )
            plot_trajectories_and_samples(
                real_data_2d=gt_2d,
                gen_data_2d=samples_2d,
                real_traj=real_traj,
                gen_traj=gen_traj,
                num_timesteps=self.num_timesteps,
                save_filepath=os.path.join(
                    self.save_dir, f"step_{step}_trajectory.png"
                ),
                proj_mat=proj_mat,
            )
