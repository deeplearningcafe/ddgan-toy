import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import Synthetic2DDataset
from trainer import DDGANTrainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="DDGAN 2D Synthetic Benchmark")
    parser.add_argument("--dataset", type=str, default="grid")
    parser.add_argument("--num_timesteps", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--iterations", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=4e-4)
    parser.add_argument("--r1_gamma", type=float, default=0.05)
    parser.add_argument("--afd_weight", type=float, default=0.5)
    parser.add_argument("--pred_target", type=str, default="x0")
    parser.add_argument("--projection_dim", type=int, default=0)
    parser.add_argument("--vanilla_gan", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./results_ddgan_grid")
    args = parser.parse_args()

    set_seed(args.seed)

    dataset = Synthetic2DDataset(
        name=args.dataset,
        n_samples=250000,
        projection_dim=args.projection_dim,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    config = {
        "dataset_name": args.dataset,
        "num_timesteps": args.num_timesteps,
        "scheduler_type": args.scheduler,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "r1_gamma": args.r1_gamma,
        "afd_weight": args.afd_weight,
        "pred_target": args.pred_target,
        "projection_dim": args.projection_dim,
        "vanilla_gan": args.vanilla_gan,
        "use_amp": args.use_amp,
        "save_path": args.save_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = DDGANTrainer(config)
    trainer.train(dataset, dataloader, args.iterations)


if __name__ == "__main__":
    main()
