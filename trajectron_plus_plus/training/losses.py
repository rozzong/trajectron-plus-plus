from typing import Optional

import torch

from ..model.mgcvae.latent.distributions import LatentDistribution


def nll_loss(
        y_dist: torch.distributions.Distribution,
        labels: torch.Tensor,
        log_p_yt_xz_max: Optional[float] = None
) -> torch.Tensor:
    """Compute the negative log-likelihood (NLL) loss.

    Parameters
    ----------
    y_dist : torch.distributions.Distribution
        Predicted trajectories distribution.
    labels : torch.Tensor
        Ground truth trajectories
    log_p_yt_xz_max : Optional[float]
        Maximum value for log-probabilities.

    Returns
    -------
    torch.Tensor
        The NLL loss.
    """
    log_p_yt_xz = y_dist.log_prob(labels)
    if log_p_yt_xz_max is not None:
        log_p_yt_xz = torch.clamp(log_p_yt_xz, max=log_p_yt_xz_max)
    log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
    log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)
    log_likelihood = torch.mean(log_p_y_xz_mean)

    return -log_likelihood


def elbo_loss(
        y_dist: torch.distributions.Distribution,
        p_dist: "LatentDistribution",
        labels: torch.Tensor,
        kl: float,
        kl_weight: float,
        log_p_yt_xz_max: Optional[float] = None
) -> torch.Tensor:
    """Compute the evidence lower bound (ELBO) loss.

    Parameters
    ----------
    y_dist : torch.distributions.Distribution
        Predicted trajectories distribution.
    p_dist : LatentDistribution
        Trajectron++ P distribution.
    labels : torch.Tensor
        Ground truth trajectories
    kl : float
        KL divergence.
    kl_weight : float
        Weight to apply on the KL divergence.
    log_p_yt_xz_max : Optional[float]
        Maximum value for log-probabilities.

    Returns
    -------
    torch.Tensor
        The ELBO loss.
    """
    nll = nll_loss(y_dist, labels, log_p_yt_xz_max)
    elbo = nll + kl_weight * kl - p_dist.mutual_inf

    return elbo
