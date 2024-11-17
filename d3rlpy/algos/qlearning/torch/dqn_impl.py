import dataclasses

import torch
from torch import nn

from ....dataclass_utils import asdict_as_float
from ....models.torch import DiscreteEnsembleQFunctionForwarder
from ....optimizers.optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    hard_sync,
)
from ....types import TorchObservation
from ..functional import ActionSampler, Updater, ValuePredictor

__all__ = [
    "DQNModules",
    "DQNLoss",
    "DQNLossFn",
    "DoubleDQNLossFn",
    "DQNActionSampler",
    "DQNActionSampler",
]


@dataclasses.dataclass(frozen=True)
class DQNModules(Modules):
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    optim: OptimizerWrapper


@dataclasses.dataclass(frozen=True)
class DQNLoss:
    loss: torch.Tensor


class DQNLossFn:
    def __init__(
        self,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        gamma: float,
    ):
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._gamma = gamma

    def __call__(self, batch: TorchMiniBatch) -> DQNLoss:
        q_tpn = self.compute_target(batch)
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )
        return DQNLoss(loss=loss)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self._targ_q_func_forwarder.compute_expected_q(
                batch.next_observations
            )
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )


class DoubleDQNLossFn(DQNLossFn):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self._q_func_forwarder.compute_expected_q(batch.next_observations).argmax(dim=1)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )


class DQNActionSampler(ActionSampler):
    def __init__(self, q_func_forwarder: DiscreteEnsembleQFunctionForwarder):
        self._q_func_forwarder = q_func_forwarder

    def __call__(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(x).argmax(dim=1)


class DQNValuePredictor(ValuePredictor):
    def __init__(self, q_func_forwarder: DiscreteEnsembleQFunctionForwarder):
        self._q_func_forwarder = q_func_forwarder

    def __call__(self, x: TorchObservation, action: torch.Tensor) -> torch.Tensor:
        values = self._q_func_forwarder.compute_expected_q(x, reduction="mean")
        flat_action = action.reshape(-1)
        return values[torch.arange(0, values.size(0)), flat_action].reshape(-1)


class DQNUpdater(Updater):
    def __init__(
        self,
        q_funcs: nn.ModuleList,
        targ_q_funcs: nn.ModuleList,
        optim: OptimizerWrapper,
        dqn_loss_fn: DQNLossFn,
        target_update_interval: int,
        compiled: bool,
    ):
        self._q_funcs = q_funcs
        self._targ_q_funcs = targ_q_funcs
        self._optim = optim
        self._dqn_loss_fn = dqn_loss_fn
        self._target_update_interval = target_update_interval
        self._compute_grad = CudaGraphWrapper(self.compute_grad) if compiled else self.compute_grad

    def compute_grad(self, batch: TorchMiniBatch) -> DQNLoss:
        self._optim.zero_grad()
        loss = self._dqn_loss_fn(batch)
        loss.loss.backward()
        return loss

    def __call__(self, batch: TorchMiniBatch, grad_step: int) -> dict[str, float]:
        loss = self._compute_grad(batch)
        self._optim.step()
        if grad_step % self._target_update_interval == 0:
            hard_sync(self._targ_q_funcs, self._q_funcs)
        return asdict_as_float(loss)
