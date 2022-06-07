from typing import Callable, Optional, Tuple, Union

import torch


def average_displacement_error(
        preds: torch.Tensor,
        targets: torch.Tensor,
        reduce: Optional[Union[str, Callable]] = None,
        index: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """Compute the average displacement error.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted trajectories. They should have the following shape:
        >>> preds.shape
        torch.Size([batch_size, n_samples, n_future_timesteps, 2])
    targets : torch.Tensor
        Ground truth trajectories. They should have the following shape:
        >>> targets.shape
        torch.Size([batch_size, n_future_timesteps, 2])
    reduce : Optional[Union[str, Callable]]
        Reduction function to apply over different samples.
    index : Optional[int]
        Index of the prediction horizon to compute the metric for.
    Returns
    -------
    torch.Tensor
        The errors for all trajectories.
    """
    # Keep the motion data up until the wanted prediction horizon
    if index is not None:
        index += 1
    preds = preds[:, :, :index]
    targets = targets[:, :index]

    # Repeat targets over samples
    targets = targets.unsqueeze(1).repeat(1, preds.shape[1], 1, 1)

    # Compute the average displacement errors
    errors = torch.linalg.norm(targets - preds, ord=2, dim=-1).mean(-1)

    # Reduce the errors over samples, if asked
    if reduce is not None:
        if isinstance(reduce, str):
            reduce = getattr(torch, reduce)
        errors = reduce(errors, dim=1)

    return errors


def final_displacement_error(
        preds: torch.Tensor,
        targets: torch.Tensor,
        reduce: Optional[Callable] = None,
        index: Optional[int] = None
) -> Tuple[torch.Tensor, int]:
    """Compute the final displacement error.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted trajectories. They should have the following shape:
        >>> preds.shape
        torch.Size([batch_size, n_samples, n_future_timesteps, 2])
    targets : torch.Tensor
        Ground truth trajectories. They should have the following shape:
        >>> targets.shape
        torch.Size([batch_size, n_future_timesteps, 2])
    reduce : Optional[Union[str, Callable]]
        Reduction function to apply over different samples.
    index : Optional[int]
        Index of the prediction horizon to compute the metric for.
    Returns
    -------
    torch.Tensor
        The errors for all trajectories.
    """
    # Keep the motion data up until the wanted prediction horizon
    if index is None:
        index = -1
    preds = preds[:, :, index]
    targets = targets[:, index]

    # Repeat targets over samples
    targets = targets.unsqueeze(1).repeat(1, preds.shape[1], 1)

    # Compute the final displacement errors
    errors = torch.linalg.norm(targets - preds, ord=2, dim=-1)

    # Reduce the errors over samples, if asked
    if reduce is not None:
        if isinstance(reduce, str):
            reduce = getattr(torch, reduce)
        errors = reduce(errors, dim=1)

    return errors
