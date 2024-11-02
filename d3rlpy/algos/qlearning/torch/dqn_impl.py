import dataclasses
from typing import Callable, Dict

import torch
from torch import nn
from torch.optim import Optimizer

from ....dataclass_utils import asdict_as_float
from ....models.torch import DiscreteEnsembleQFunctionForwarder
from ....optimizers.optimizers import OptimizerWrapper
from ....torch_utility import (
    CudaGraphWrapper,
    Modules,
    TorchMiniBatch,
    hard_sync,
)
from ....types import Shape, TorchObservation
from ..base import QLearningAlgoImplBase
from .utility import DiscreteQFunctionMixin

__all__ = ["DQNImpl", "DQNModules", "DQNLoss", "DoubleDQNImpl"]


@dataclasses.dataclass(frozen=True)
class DQNModules(Modules):
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    optim: OptimizerWrapper


@dataclasses.dataclass(frozen=True)
class DQNLoss:
    loss: torch.Tensor


class DQNImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _modules: DQNModules
    _compute_grad: Callable[[TorchMiniBatch], DQNLoss]
    _gamma: float
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _target_update_interval: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DQNModules,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        target_update_interval: int,
        gamma: float,
        compile_graph: bool,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        self._compute_grad = (
            CudaGraphWrapper(self.compute_grad)  # type: ignore
            if compile_graph
            else self.compute_grad
        )
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_grad(self, batch: TorchMiniBatch) -> DQNLoss:
        self._modules.optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_loss(batch, q_tpn)
        loss.loss.backward()
        return loss

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        loss = self._compute_grad(batch)
        self._modules.optim.step()
        if grad_step % self._target_update_interval == 0:
            self.update_target()
        return asdict_as_float(loss)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> DQNLoss:
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

    def inner_predict_best_action(self, x: TorchObservation) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(x).argmax(dim=1)

    def inner_sample_action(self, x: TorchObservation) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    @property
    def q_function(self) -> nn.ModuleList:
        return self._modules.q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._modules.optim.optim


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
