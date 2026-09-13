# DDGAN-Toy: Denoising Diffusion GANs on 2D Synthetic Manifolds

A modular, lightweight sandbox designed to study and reproduce the synthetic manifold benchmarks from the ICLR 2022 paper [**"Tackling the Generative Learning Trilemma with Denoising Diffusion GANs"**](https://arxiv.org/abs/2112.07804) (Xiao et al.), with a focus on **Figure 6 (the 25-Gaussians grid)** and complex multimodal 2D toy distributions.

---

## Table of Contents

- [Overview](#overview)
- [Key Features & Theoretical Foundations](#key-features--theoretical-foundations)
- [Repository Structure](#repository-structure)
- [Prerequisites & Requirements](#prerequisites--requirements)
- [Installation](#installation)
- [Workflow and Usage](#workflow-and-usage)
  - [1. Reproducing Figure 6 (25-Gaussians Grid)](#1-reproducing-figure-6-25-gaussians-grid)
  - [2. Vanilla GAN Baseline (Mode Collapse Demonstration)](#2-vanilla-gan-baseline-mode-collapse-demonstration)
  - [3. Long-Tail & Imbalanced Density Benchmarks](#3-long-tail--imbalanced-density-benchmarks)
  - [4. High-Dimensional Manifold Projection ($D > 2$)](#4-high-dimensional-manifold-projection-d--2)
  - [5. Parametrization Ablation ($\hat{x}_0$ vs. $\epsilon$ vs. $v$)](#5-parametrization-ablation-hatx_0-vs-epsilon-vs-v)
- [Evaluation & Quantitative Metrics](#evaluation--quantitative-metrics)
- [Hardware & Precision Configuration](#hardware--precision-configuration)
- [References](#references)
- [Author & License](#author--license)

---

## Overview

Deep generative modeling faces the **generative learning trilemma**: achieving (1) high sample fidelity, (2) fast inference, and (3) full mode coverage simultaneously.
* **Standard Diffusion Models (DDPM / SDEs):** Achieve high quality and faithful mode coverage, but require tens to hundreds of sequential iterations due to the infinitesimal Gaussian reverse-step assumption.
* **Traditional GANs:** Fast and sharp in a single shot, but suffer from catastrophic mode collapse and training instability.
* **Denoising Diffusion GANs (DD-GAN):** Parametrize the reverse denoising distribution $q(x_t \mid x_{t+1})$ with a conditional GAN generator $G_\theta(x_{t+1}, t, z) \to \hat{x}_0$ driven by a latent variable $z \sim \mathcal{N}(0, I)$. This permits large time-step jumps ($T = 4$), enabling full mode coverage without collapsing modes.

```
                  Infinitesimal Step: q(x_{t-1} | x_t) is unimodal Gaussian
DDPM:    x_T --------------> x_{T-1} --------------> ... --------------> x_0  (T = 1000)

                  Large Step: q(x_t | x_{t+1}) is multimodal non-Gaussian
DD-GAN:  x_T ======================> x_t ======================> x_0          (T = 4)
                      G_theta(x_{t+1}, t, z) + Posterior Sampling
```

This repository isolates the 2D dynamics into a clean, hackable codebase with PyTorch `DataLoader` pipelines, Tensor Core acceleration, quantitative metric logging (Chamfer Distance, Precision, Recall, Curvature), and co-located diagnostic plots.

---

## Key Features & Theoretical Foundations

### 1. Multimodal Reverse Step via Posterior Mapping
Rather than directly regressing $x_t$, the generator predicts the clean target $\hat{x}_0 = G_\theta(x_{t+1}, t, z)$. Reverse transitions are drawn analytically via the tractable Gaussian posterior:
$$q(x_t \mid x_{t+1}, \hat{x}_0) = \mathcal{N}\left(x_t;\, \tilde{\mu}_t(x_{t+1}, \hat{x}_0),\, \tilde{\beta}_t \mathbf{I}\right)$$
This decouples the multimodality (handled by the latent variable $z$) from the intermediate noise injection.

### 2. $R_1$ Zero-Centered Gradient Penalty
To guarantee local Nash equilibrium and prevent discriminator gradient explosion on large step sizes, we compute the explicit $R_1$ penalty on real transitions:
$$R_1(\phi) = \frac{\gamma}{2} \, \mathbb{E}_{q(x_t, x_{t+1})}\left[ \|\nabla_{x_t} D_\phi(x_t, x_{t+1}, t)\|_2^2 \right]$$

### 3. Auxiliary Forward Diffusion (AFD) Cycle-Consistency
To prevent the generator from proposing plausible but unfaithful reverse trajectories, the model incorporates the AFD cycle-consistency penalty:
$$\mathcal{L}_{\text{AFD}} = \mathbb{E}\left[ \| q(x_{t+1} \mid x_t^{\text{fake}}) - x_{t+1} \|_2^2 \right]$$
Re-diffusing the synthesized $x_t^{\text{fake}}$ forward by one step must reconstruct the conditioning state $x_{t+1}$.

---

## Repository Structure

```
ddgan_toy/
├── data/
│   ├── __init__.py
│   └── dataset.py           # 2D synthetic manifolds (Grid, Circle, Spiral, etc.)
├── evaluation.py            # Chamfer Distance, Precision/Recall, and Path Curvature
├── losses.py                # Adversarial non-saturating loss, R1 penalty, and AFD cycle loss
├── models/
│   ├── __init__.py
│   └── mlp.py               # GeneratorND, DiscriminatorND, SinusoidalTimeEmbedding
├── paths/
│   ├── __init__.py
│   ├── sampling.py          # Reverse SDE sampler, posterior sampling, Markov transitions
│   └── scheduler.py         # Diffusion schedules: VP-DDPM, Linear (Flow-style), Cosine
├── scripts/
│   └── train.py             # CLI entry point for training and evaluation
├── trainer.py               # DDGANTrainer class with step loops and evaluation hooks
└── utils/
    ├── __init__.py
    ├── logging_utils.py     # Dual console and disk text logging to run_log.txt
    ├── trainer_utils.py     # Hardware detection, Tensor Core/AMP setup, TTUR optimizers
    └── visualization.py     # Scatter + KDE density maps, forward diagnostics, trajectories
```

---

## Prerequisites & Requirements

* **OS:** Linux (Ubuntu 20.04+ recommended) or macOS
* **Python:** >= 3.10
* **CUDA Hardware:** NVIDIA GPU


### Dependencies (`requirements.txt`)
```txt
torch>=2.4.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/deeplearningcafe/ddgan-toy.git
cd ddgan-toy

# 2. Create and activate a clean environment
python -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

```

---

## Workflow and Usage

All experiments are executed via `ddgan_toy/scripts/train.py`. Results, checkpoints, diagnostic plots, and execution logs (`run_log.txt`) are automatically co-located in the designated `--save_path`.

### 1. Reproducing Figure 6 (25-Gaussians Grid)
Train a 4-step DD-GAN on a $5 \times 5$ Gaussian mixture grid to verify complete mode coverage without collapsed clusters:
```bash
python -m scripts.train \
    --dataset grid \
    --num_timesteps 4 \
    --scheduler linear \
    --iterations 50000 \
    --batch_size 512 \
    --lr_g 0.0001 \
    --lr_d 0.0004 \
    --r1_gamma 0.05 \
    --afd_weight 0.5 \
    --pred_target x0 \
    --save_path ./results/figure6_ddgan_grid
```

### 2. Vanilla GAN Baseline (Mode Collapse Demonstration)
Train an unconditional single-step vanilla GAN on the exact same dataset to observe classical mode dropping:
```bash
python -m scripts.train \
    --dataset grid \
    --vanilla_gan \
    --iterations 50000 \
    --batch_size 512 \
    --lr_g 0.0001 \
    --lr_d 0.0004 \
    --r1_gamma 0.05 \
    --save_path ./results/baseline_vanilla_gan
```

### 3. Long-Tail & Imbalanced Density Benchmarks
Evaluate performance under extreme probability imbalances (e.g., 10 modes along the x-axis with exponentially decaying probabilities $p_i \propto 2^{-(i+1)}$):
```bash
python -m scripts.train \
    --dataset gmm_long_tail \
    --num_timesteps 4 \
    --scheduler linear \
    --iterations 50000 \
    --afd_weight 0.5 \
    --save_path ./results/gmm_long_tail_linear
```

### 4. High-Dimensional Manifold Projection ($D > 2$)
To test stability on high-dimensional data embedded along lower-dimensional manifolds, set `--projection_dim` to project 2D data into $N$-dimensional space via a random orthogonal matrix ($P \in \mathbb{R}^{D \times 2}$):
```bash
python -m scripts.train \
    --dataset pinwheel \
    --projection_dim 64 \
    --num_timesteps 4 \
    --iterations 50000 \
    --save_path ./results/pinwheel_projected_64d
```

### 5. Parametrization Ablation ($\hat{x}_0$ vs. $\epsilon$ vs. $v$)
Compare clean data prediction against noise or velocity prediction targets:
```bash
# Epsilon prediction
python -m scripts.train \
    --dataset grid \
    --pred_target eps \
    --save_path ./results/grid_pred_eps

# Velocity (v) prediction
python -m scripts.train \
    --dataset grid \
    --pred_target v \
    --save_path ./results/grid_pred_v
```

---

## Evaluation & Quantitative Metrics

Every `--eval_interval` steps (default: `5000`), the trainer executes evaluation on 5,000 generated points and logs the output directly to the console and `run_log.txt`:

1. **Symmetric Chamfer Distance:** Measures spatial displacement between generated samples and ground truth data:
   $$\mathcal{D}_{\text{Chamfer}}(\mathcal{S}_1, \mathcal{S}_2) = \frac{1}{|\mathcal{S}_1|} \sum_{x \in \mathcal{S}_1} \min_{y \in \mathcal{S}_2} \|x - y\|_2^2 + \frac{1}{|\mathcal{S}_2|} \sum_{y \in \mathcal{S}_2} \min_{x \in \mathcal{S}_1} \|y - x\|_2^2$$
2. **Manifold Precision & Recall ($k$-NN):** Evaluates fidelity (Precision) and coverage (Recall) using $k$-nearest-neighbor hyperspheres on the data manifold ($k=5$).
3. **Path Curvature:** Measures trajectory straightness across the $T$-step reverse denoising process:
   $$\text{Curvature} = \frac{\sum_{t=1}^T \|x_{t-1} - x_t\|_2}{\|x_0 - x_T\|_2}$$
   *(A value of $1.0$ represents a completely straight path).*

### Generated Diagnostic Artifacts
* `forward_diffusion_evolution.png`: Signal-to-noise ratio degradation curves and ground-truth forward process states.
* `step_{step}.png`: Dual-panel scatter plot and 2D Kernel Density Estimation (KDE) contour overlay.
* `step_{step}_trajectory.png`: Comparative transition phase portraits contrasting real forward transitions with generator denoising transitions.
* `run_log.txt`: Timestamped iteration losses ($D$, $G$, $R_1$, $\text{AFD}$), gradient norms, and quantitative evaluation scores.

---


## References

* **Denoising Diffusion GANs Paper:**
  Xiao, Z., Kreis, K., & Vahdat, A. (2022). *Tackling the Generative Learning Trilemma with Denoising Diffusion GANs*. International Conference on Learning Representations (ICLR 2022). [arXiv:2112.07804](https://arxiv.org/abs/2112.07804).
* **Base Repository:**
  [Toy-Diffusion](https://github.com/deeplearningcafe/toy-diffusion): Exploration for Flow Matching, EDM, and Consistency Models.

---

## Author & License

* **Author:** [aipracticecafe](https://github.com/deeplearningcafe) ([Codeberg](https://codeberg.org/aipracticecafe))
* **License:** This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
