import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from trainer import DDGANTrainer
from utils.trainer_utils import set_seed


def main():
    parser = argparse.ArgumentParser(description="DDGAN MNIST Benchmark")
    parser.add_argument("--num_timesteps", type=int, default=4)
    parser.add_argument("--scheduler", type=str, default="ddpm")
    parser.add_argument("--iterations", type=int, default=60000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_g", type=float, default=1.5e-4)
    parser.add_argument("--lr_d", type=float, default=1e-4)
    parser.add_argument("--r1_gamma", type=float, default=0.05)
    parser.add_argument("--afd_weight", type=float, default=0.0)
    parser.add_argument("--pred_target", type=str, default="x0")
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--ch", type=int, default=64)
    parser.add_argument("--vanilla_gan", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="./results_ddgan_mnist")
    args = parser.parse_args()

    set_seed(args.seed)

    # Resize to 32x32 to allow symmetric powers-of-two downsampling
    transform = transforms.Compose(
        [
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
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
        "dataset_name": "mnist",
        "is_image": True,
        "data_shape": (1, 32, 32),
        "z_dim": args.z_dim,
        "hidden_dim": args.ch,
        "num_timesteps": args.num_timesteps,
        "scheduler_type": args.scheduler,
        "iterations": args.iterations,
        "batch_size": args.batch_size,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "r1_gamma": args.r1_gamma,
        "afd_weight": args.afd_weight,
        "pred_target": args.pred_target,
        "vanilla_gan": args.vanilla_gan,
        "use_amp": args.use_amp,
        "save_path": args.save_path,
        "eval_interval": 2000,
        "log_interval": 100,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    trainer = DDGANTrainer(config)
    trainer.train(dataset, dataloader, args.iterations)


if __name__ == "__main__":
    main()
