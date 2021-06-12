from typing import cast

import torch
import torch.nn.functional as F


def pick_value_by_action(
    values: torch.Tensor, action: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    assert values.ndim == 2
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    masked_values = values * cast(torch.Tensor, one_hot.float())
    return masked_values.sum(dim=1, keepdim=keepdim)


def pick_quantile_value_by_action(
    values: torch.Tensor, action: torch.Tensor, keepdim: bool = False
) -> torch.Tensor:
    assert values.ndim == 3
    action_size = values.shape[1]
    one_hot = F.one_hot(action.view(-1), num_classes=action_size)
    mask = cast(torch.Tensor, one_hot.view(-1, action_size, 1).float())
    return (values * mask).sum(dim=1, keepdim=keepdim)


def compute_huber_loss(
    y: torch.Tensor, target: torch.Tensor, beta: float = 1.0
) -> torch.Tensor:
    diff = target - y
    cond = diff.detach().abs() < beta
    return torch.where(cond, 0.5 * diff ** 2, beta * (diff.abs() - 0.5 * beta))


def compute_quantile_huber_loss(
    y: torch.Tensor, target: torch.Tensor, taus: torch.Tensor
) -> torch.Tensor:
    assert y.dim() == 3 and target.dim() == 3 and taus.dim() == 3
    # compute huber loss
    huber_loss = compute_huber_loss(y, target)
    delta = cast(torch.Tensor, ((target - y).detach() < 0.0).float())
    element_wise_loss = (taus - delta).abs() * huber_loss
    return element_wise_loss.sum(dim=2).mean(dim=1)


def compute_reduce(value: torch.Tensor, reduction_type: str) -> torch.Tensor:
    if reduction_type == "mean":
        return value.mean()
    elif reduction_type == "sum":
        return value.sum()
    elif reduction_type == "none":
        return value.view(-1, 1)
    raise ValueError("invalid reduction type.")
