"""Timestep samplers for diffusion training.

Vendored from stable-audio-tools/inference/sampling.py — these definitions
are identical in both backends, so vendoring lets the loop be backend-free.
"""
import math

import torch
import torch.distributions as dist


def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):
    """Sample t from a Gaussian on logSNR (Eq. logsnr = ln((1-t)/t)) → t = sigmoid(-logsnr)."""
    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr
    return torch.sigmoid(-logsnr).clamp(1e-4, 1 - 1e-4)


def sample_timesteps_logsnr_uniform(batch_size, min_logsnr=-6.0, max_logsnr=5.0):
    """Sample t from a uniform on logSNR."""
    logsnr = torch.rand(batch_size) * (max_logsnr - min_logsnr) + min_logsnr
    return torch.sigmoid(-logsnr).clamp(1e-4, 1 - 1e-4)


def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1.0):
    """Truncated logistic-normal, rescaled to [0, 1)."""
    logits = torch.randn(shape)
    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)
    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))
    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values
    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))
    return (truncated_samples - left_trunc) / (right_trunc - left_trunc)


# Anchor for the timestep-ratio feature: the median of the un-shifted
# trunc_logit_normal output, measured over 2M samples. ratio = 0.5 maps to this
# center, keeping the default bit-identical to the original sampler.
_TLN_LEFT_TRUNC = 0.075
_TLN_CENTER = 0.5385


def _tln_z_of_center(c):
    """Map a timestep center c in (0,1) to its z (logit) coordinate: the z such
    that t(z) = c, where t = 1 - (sigmoid(z) - left)/(1 - left)."""
    p = _TLN_LEFT_TRUNC + (1.0 - _TLN_LEFT_TRUNC) * (1.0 - c)
    return math.log(p / (1.0 - p))


def sample_t_tln_shifted(batch_size, device, ratio):
    """trunc_logit_normal timesteps with a shape-preserving center shift.

    `ratio` (0..1) sets the timestep the distribution's center should sit at
    (0 = pure signal / high-SNR, 1 = pure noise / low-SNR): ratio < 0.5 biases
    toward LOW timesteps, ratio > 0.5 toward HIGH timesteps. Passing None (the
    toggle off) or 0.5 reproduces the original `trunc_logit_normal` sampler
    exactly; the bell width is unchanged - only the mean moves.
    """
    if ratio is None:
        return (1 - truncated_logistic_normal_rescaled(batch_size)).to(device)
    ratio = float(ratio)
    if abs(ratio - 0.5) < 1e-9:
        return (1 - truncated_logistic_normal_rescaled(batch_size)).to(device)
    target_center = min(max(ratio + (_TLN_CENTER - 0.5), 0.05), 0.95)
    delta = _tln_z_of_center(target_center) - _tln_z_of_center(_TLN_CENTER)
    # z ~ N(delta, 1) | z > logit(left_trunc): the original sampler's one-sided
    # truncation, with only the mean moved by `delta`.
    normal_dist = dist.Normal(0, 1)
    logits = torch.randn(batch_size)
    cdf_values = normal_dist.cdf(logits)
    a = math.log(_TLN_LEFT_TRUNC / (1.0 - _TLN_LEFT_TRUNC))
    lower_bound = normal_dist.cdf(torch.tensor(a - delta))
    truncated_cdf_values = lower_bound + (1.0 - lower_bound) * cdf_values
    z = normal_dist.icdf(truncated_cdf_values) + delta
    truncated_samples = torch.sigmoid(z)
    return (1 - (truncated_samples - _TLN_LEFT_TRUNC) / (1.0 - _TLN_LEFT_TRUNC)).to(device)


def sample_t(timestep_sampler, batch_size, device, options=None):
    """Dispatch to the configured timestep sampler.

    `timestep_sampler` is one of: "uniform", "logit_normal", "trunc_logit_normal",
    "log_snr", "log_snr_uniform". `options` is an optional dict (currently used
    only by log_snr* variants for {mean,std}_logsnr / {min,max}_logsnr).
    """
    options = options or {}
    if timestep_sampler == "uniform":
        t = torch.rand(batch_size, device=device)
    elif timestep_sampler == "logit_normal":
        t = torch.sigmoid(torch.randn(batch_size, device=device))
    elif timestep_sampler == "trunc_logit_normal":
        # trunc + flip to match SAT-dev. options["ratio"] (0..1) shifts the bell
        # center toward low (<0.5) or high (>0.5) timesteps; None/0.5 = original.
        t = sample_t_tln_shifted(batch_size, device, options.get("ratio"))
    elif timestep_sampler == "log_snr":
        t = sample_timesteps_logsnr(
            batch_size,
            mean_logsnr=options.get("mean_logsnr", -1.2),
            std_logsnr=options.get("std_logsnr", 2.0),
        ).to(device)
    elif timestep_sampler == "log_snr_uniform":
        t = sample_timesteps_logsnr_uniform(
            batch_size,
            min_logsnr=options.get("min_logsnr", -6.0),
            max_logsnr=options.get("max_logsnr", 5.0),
        ).to(device)
    else:
        raise ValueError(f"Invalid timestep_sampler: {timestep_sampler}")
    return t
