from dataclasses import replace
from typing import Optional, Sequence, Union

import torch

from ....torch_utility import get_device
from ....types import TorchObservation
from .base import (
    ContinuousQFunctionForwarder,
    DiscreteQFunctionForwarder,
    TargetOutput,
)
from .iqn_q_function import ImplicitQuantileTargetOutput
from .qr_q_function import QuantileTargetOutput

__all__ = [
    "DiscreteEnsembleQFunctionForwarder",
    "ContinuousEnsembleQFunctionForwarder",
    "compute_max_with_n_actions",
    "compute_max_with_n_actions_and_indices",
]


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices.view(-1)]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor,
    stacked_q_values: torch.Tensor,
    reduction: str = "min",
    dim: int = 0,
    lam: float = 0.75,
) -> torch.Tensor:
    if reduction == "min":
        indices = stacked_q_values.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = stacked_q_values.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = stacked_q_values.min(dim=dim).indices
        max_indices = stacked_q_values.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _reduce_target_outputs(
    target_outputs: list[TargetOutput],
    reduction: str = "min",
    dim: int = 0,
    lam: float = 0.75,
) -> TargetOutput:
    stacked_values = torch.stack([q.q_value for q in target_outputs], dim=0)
    reduced_value = _reduce_ensemble(
        y=stacked_values,
        reduction=reduction,
        dim=dim,
        lam=lam,
    )
    return TargetOutput(reduced_value)


def _reduce_quantile_target_outputs(
    target_outputs: list[QuantileTargetOutput],
    reduction: str = "min",
    dim: int = 0,
    lam: float = 0.75,
) -> QuantileTargetOutput:
    stacked_values = torch.stack([q.q_value for q in target_outputs], dim=0)
    stacked_quantiles = torch.stack([q.quantile for q in target_outputs], dim=0)
    reduced_value = _reduce_ensemble(
        y=stacked_values,
        reduction=reduction,
        dim=dim,
        lam=lam,
    )
    reduced_quantile = _reduce_quantile_ensemble(
        y=stacked_quantiles,
        stacked_q_values=stacked_values,
        reduction=reduction,
        dim=dim,
        lam=lam,
    )
    return QuantileTargetOutput(
        q_value=reduced_value,
        quantile=reduced_quantile,
    )


def _reduce_implicit_quantile_target_outputs(
    target_outputs: list[ImplicitQuantileTargetOutput],
    reduction: str = "min",
    dim: int = 0,
    lam: float = 0.75,
) -> ImplicitQuantileTargetOutput:
    stacked_values = torch.stack([q.q_value for q in target_outputs], dim=0)
    stacked_quantiles = torch.stack([q.quantile for q in target_outputs], dim=0)
    stacked_taus = torch.stack([q.taus for q in target_outputs], dim=0)
    reduced_value = _reduce_ensemble(
        y=stacked_values,
        reduction=reduction,
        dim=dim,
        lam=lam,
    )
    reduced_quantile = _reduce_quantile_ensemble(
        y=stacked_quantiles,
        stacked_q_values=stacked_values,
        reduction=reduction,
        dim=dim,
        lam=lam,
    )
    if stacked_values.shape[-1] == 1:
        reduced_taus = _reduce_quantile_ensemble(
            y=stacked_taus,
            stacked_q_values=stacked_values,
            reduction=reduction,
            dim=dim,
            lam=lam,
        )
    else:
        action_size = stacked_values.shape[-1]
        reduced_taus = _reduce_quantile_ensemble(
            y=torch.unsqueeze(stacked_taus, dim=-2).tile(
                [1, 1, action_size, 1]
            ),
            stacked_q_values=stacked_values,
            reduction=reduction,
            dim=dim,
            lam=lam,
        )
    return ImplicitQuantileTargetOutput(
        q_value=reduced_value,
        quantile=reduced_quantile,
        taus=reduced_taus,
    )


def _reduce_generic_target_outputs(
    target_outputs: list[TargetOutput],
    reduction: str = "min",
    dim: int = 0,
    lam: float = 0.75,
) -> TargetOutput:
    if isinstance(target_outputs[0], QuantileTargetOutput):
        return _reduce_quantile_target_outputs(
            target_outputs=target_outputs,  # type: ignore
            reduction=reduction,
            dim=dim,
            lam=lam,
        )
    elif isinstance(target_outputs[0], ImplicitQuantileTargetOutput):
        return _reduce_implicit_quantile_target_outputs(
            target_outputs=target_outputs,  # type: ignore
            reduction=reduction,
            dim=dim,
            lam=lam,
        )
    elif isinstance(target_outputs[0], TargetOutput):
        return _reduce_target_outputs(
            target_outputs=target_outputs,
            reduction=reduction,
            dim=dim,
            lam=lam,
        )
    else:
        raise ValueError(f"invalid type: {type(target_outputs[0])}")


def compute_ensemble_q_function_error(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    observations: TorchObservation,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    target: TargetOutput,
    terminals: torch.Tensor,
    gamma: Union[float, torch.Tensor] = 0.99,
    masks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert target.q_value.ndim == 2
    td_sum = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=get_device(observations),
    )
    for forwarder in forwarders:
        loss = forwarder.compute_error(
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
            reduction="none",
        )
        if masks is not None:
            loss = loss * masks
        td_sum += loss.mean()
    return td_sum


def compute_ensemble_q_function_target(
    forwarders: Union[
        Sequence[DiscreteQFunctionForwarder],
        Sequence[ContinuousQFunctionForwarder],
    ],
    x: TorchObservation,
    action: Optional[torch.Tensor] = None,
    reduction: str = "min",
    lam: float = 0.75,
) -> TargetOutput:
    targets: list[TargetOutput] = []
    for forwarder in forwarders:
        if isinstance(forwarder, ContinuousQFunctionForwarder):
            assert action is not None
            target = forwarder.compute_target(x, action)
        else:
            target = forwarder.compute_target(x, action)
        targets.append(target)
    return _reduce_generic_target_outputs(
        target_outputs=targets,
        reduction=reduction,
        lam=lam,
    )


class DiscreteEnsembleQFunctionForwarder:
    _forwarders: Sequence[DiscreteQFunctionForwarder]
    _action_size: int

    def __init__(
        self, forwarders: Sequence[DiscreteQFunctionForwarder], action_size: int
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: TorchObservation, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x)
            values.append(
                value.view(
                    1,
                    (
                        x[0].shape[0]
                        if isinstance(x, (list, tuple))
                        else x.shape[0]  # type: ignore
                    ),
                    self._action_size,
                )
            )
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_error(
            forwarders=self._forwarders,
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
            masks=masks,
        )

    def compute_target(
        self,
        x: TorchObservation,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> TargetOutput:
        return compute_ensemble_q_function_target(
            forwarders=self._forwarders,
            x=x,
            action=action,
            reduction=reduction,
            lam=lam,
        )

    @property
    def forwarders(self) -> Sequence[DiscreteQFunctionForwarder]:
        return self._forwarders


class ContinuousEnsembleQFunctionForwarder:
    _forwarders: Sequence[ContinuousQFunctionForwarder]
    _action_size: int

    def __init__(
        self,
        forwarders: Sequence[ContinuousQFunctionForwarder],
        action_size: int,
    ):
        self._forwarders = forwarders
        self._action_size = action_size

    def compute_expected_q(
        self, x: TorchObservation, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for forwarder in self._forwarders:
            value = forwarder.compute_expected_q(x, action)
            values.append(
                value.view(
                    1,
                    (
                        x[0].shape[0]
                        if isinstance(x, (list, tuple))
                        else x.shape[0]  # type: ignore
                    ),
                    1,
                )
            )
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(
        self,
        observations: TorchObservation,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: TargetOutput,
        terminals: torch.Tensor,
        gamma: Union[float, torch.Tensor] = 0.99,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return compute_ensemble_q_function_error(
            forwarders=self._forwarders,
            observations=observations,
            actions=actions,
            rewards=rewards,
            target=target,
            terminals=terminals,
            gamma=gamma,
            masks=masks,
        )

    def compute_target(
        self,
        x: TorchObservation,
        action: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> TargetOutput:
        return compute_ensemble_q_function_target(
            forwarders=self._forwarders,
            x=x,
            action=action,
            reduction=reduction,
            lam=lam,
        )

    @property
    def forwarders(self) -> Sequence[ContinuousQFunctionForwarder]:
        return self._forwarders


def compute_max_with_n_actions_and_indices(
    x: TorchObservation,
    actions: torch.Tensor,
    forwarder: ContinuousEnsembleQFunctionForwarder,
    lam: float,
) -> tuple[TargetOutput, torch.Tensor]:
    """Returns weighted target value from sampled actions.

    This calculation is proposed in BCQ paper for the first time.
    `x` should be shaped with `(batch, dim_obs)`.
    `actions` should be shaped with `(batch, N, dim_action)`.
    """
    batch_size = actions.shape[0]
    n_critics = len(forwarder.forwarders)
    n_actions = actions.shape[1]

    if isinstance(x, torch.Tensor):
        # (batch, observation) -> (batch, n, observation)
        expanded_x = x.expand(n_actions, *x.shape).transpose(0, 1)
        # (batch * n, observation)
        flat_x = expanded_x.reshape(-1, *x.shape[1:])
    else:
        # (batch, observation) -> (batch, n, observation)
        expanded_x = [
            _x.expand(n_actions, *_x.shape).transpose(0, 1) for _x in x
        ]
        # (batch * n, observation)
        flat_x = [_x.reshape(-1, *_x.shape[2:]) for _x in expanded_x]
    # (batch, n, action) -> (batch * n, action)
    flat_actions = actions.reshape(batch_size * n_actions, -1)

    # estimate values while taking care of quantiles
    target_output = forwarder.compute_target(flat_x, flat_actions, "none")
    flat_values = target_output.q_value
    # reshape to (n_ensembles, batch_size, n, -1)
    transposed_values = flat_values.view(n_critics, batch_size, n_actions, -1)
    # (n_ensembles, batch_size, n, -1) -> (batch_size, n_ensembles, n, -1)
    values = transposed_values.transpose(0, 1)

    # get combination indices
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n_ensembles, n)
    mean_values = values.mean(dim=3)
    # (batch_size, n_ensembles, n) -> (batch_size, n)
    max_values, max_indices = mean_values.max(dim=1)
    min_values, min_indices = mean_values.min(dim=1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch_size, n) -> (batch_size,)
    action_indices = mix_values.argmax(dim=1)

    # fuse maximum values and minimum values
    # (batch_size, n_ensembles, n, -1) -> (batch_size, n, n_ensembles, -1)
    values_T = values.transpose(1, 2)
    # (batch, n, n_ensembles, -1) -> (batch * n, n_ensembles, -1)
    flat_values = values_T.reshape(batch_size * n_actions, n_critics, -1)
    # (batch * n, n_ensembles, -1) -> (batch * n, -1)
    bn_indices = torch.arange(batch_size * n_actions)
    max_values = flat_values[bn_indices, max_indices.view(-1)]
    min_values = flat_values[bn_indices, min_indices.view(-1)]
    # (batch * n, -1) -> (batch, n, -1)
    max_values = max_values.view(batch_size, n_actions, -1)
    min_values = min_values.view(batch_size, n_actions, -1)
    mix_values = (1.0 - lam) * max_values + lam * min_values
    # (batch, n, -1) -> (batch, -1)
    result_values = mix_values[torch.arange(batch_size), action_indices]

    return replace(target_output, q_value=result_values), action_indices


def compute_max_with_n_actions(
    x: TorchObservation,
    actions: torch.Tensor,
    forwarder: ContinuousEnsembleQFunctionForwarder,
    lam: float,
) -> TargetOutput:
    return compute_max_with_n_actions_and_indices(x, actions, forwarder, lam)[0]
