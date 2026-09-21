import torch


def _extract(coeffs: torch.Tensor, t: torch.Tensor, shape: torch.Size):
    """Extracts coefficients and reshapes for arbitrary input dimensions."""
    val = coeffs[t]
    reshape_dims = [shape[0]] + [1] * (len(shape) - 1)
    return val.view(*reshape_dims)


def forward_diffuse_step(fwd_coeffs, x_t: torch.Tensor, t: torch.Tensor):
    """Simulates one Markov forward step q(x_{t+1} | x_t)."""
    a_step = _extract(fwd_coeffs.a_s, t + 1, x_t.shape)
    s_step = _extract(fwd_coeffs.sigmas, t + 1, x_t.shape)
    return a_step * x_t + s_step * torch.randn_like(x_t)


def q_sample_pairs(fwd_coeffs, x_start: torch.Tensor, t: torch.Tensor):
    """Samples true consecutive pair (x_t, x_{t+1}) from x_0."""
    noise_t = torch.randn_like(x_start)
    a_t = _extract(fwd_coeffs.a_s_cum, t, x_start.shape)
    s_t = _extract(fwd_coeffs.sigmas_cum, t, x_start.shape)
    x_t = a_t * x_start + s_t * noise_t

    noise_step = torch.randn_like(x_start)
    a_tp1 = _extract(fwd_coeffs.a_s, t + 1, x_start.shape)
    s_tp1 = _extract(fwd_coeffs.sigmas, t + 1, x_start.shape)
    x_tp1 = a_tp1 * x_t + s_tp1 * noise_step
    return x_t, x_tp1


def sample_posterior(pos_coeffs, x_0: torch.Tensor, x_tp1: torch.Tensor, t):
    """Draws x_t ~ q(x_t | x_{t+1}, x_0). Zero noise added at t=0."""
    c1 = _extract(pos_coeffs.coef1, t, x_tp1.shape)
    c2 = _extract(pos_coeffs.coef2, t, x_tp1.shape)
    mean = c1 * x_0 + c2 * x_tp1
    var = _extract(pos_coeffs.var, t, x_tp1.shape)
    nonzero_mask = _extract((t != 0).float(), t, x_tp1.shape)
    return mean + nonzero_mask * torch.sqrt(var) * torch.randn_like(x_tp1)


def convert_pred_to_x0(
    pred: torch.Tensor,
    x_tp1: torch.Tensor,
    fwd_coeffs,
    t: torch.Tensor,
    pred_target: str = "x0",
) -> torch.Tensor:
    """Converts prediction to clean estimate x0 for posterior sampling."""
    if pred_target == "x0":
        return pred

    a_cum = _extract(fwd_coeffs.a_s_cum, t + 1, x_tp1.shape).clamp(min=1e-5)
    s_cum = _extract(fwd_coeffs.sigmas_cum, t + 1, x_tp1.shape)

    if pred_target == "eps":
        return (x_tp1 - s_cum * pred) / a_cum
    elif pred_target == "v":
        return a_cum * x_tp1 - s_cum * pred
    raise ValueError(f"Unknown prediction target: {pred_target}")


@torch.no_grad()
def sample_from_model(
    pos_coeffs,
    fwd_coeffs,
    net_g,
    num_timesteps: int,
    num_samples: int,
    data_dim: int,
    z_dim: int,
    device: torch.device,
    pred_target: str = "x0",
    vanilla_gan: bool = False,
    return_trajectory: bool = False,
):
    """Executes iterative reverse denoising using the generator and EMA."""
    net_g.eval()
    trajectory = []
    spatial_shape = (data_dim,) if isinstance(data_dim, int) else tuple(data_dim)

    if vanilla_gan:
        z = torch.randn(num_samples, z_dim, device=device)
        samples = net_g(None, None, z)
        if return_trajectory:
            trajectory = [samples]
    else:
        x = torch.randn(num_samples, *spatial_shape, device=device)
        if return_trajectory:
            trajectory.append(x.clone())

        for i in reversed(range(num_timesteps)):
            t = torch.full((num_samples,), i, dtype=torch.long, device=device)
            z = torch.randn(num_samples, z_dim, device=device)
            raw_pred = net_g(x, t, z)
            x_0_pred = convert_pred_to_x0(
                raw_pred, x, fwd_coeffs, t, pred_target=pred_target
            )
            x = sample_posterior(pos_coeffs, x_0_pred, x, t)
            if return_trajectory:
                trajectory.append(x.clone())

        samples = x

    net_g.train()
    if return_trajectory:
        return samples, trajectory
    return samples


def get_real_forward_trajectories(
    fwd_coeffs, x_0_batch: torch.Tensor, num_timesteps: int
):
    """Computes exact Markov forward states for diagnostic comparisons."""
    states = [x_0_batch]
    curr_x = x_0_batch
    for step in range(num_timesteps):
        t_idx = torch.full(
            (curr_x.size(0),), step, dtype=torch.long, device=curr_x.device
        )
        curr_x = forward_diffuse_step(fwd_coeffs, curr_x, t_idx)
        states.append(curr_x)
    return states
