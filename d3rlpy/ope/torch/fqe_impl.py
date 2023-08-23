import dataclasses
from typing import Dict, Union

import torch
from torch import nn
from torch.optim import Optimizer

from ...algos.qlearning.base import QLearningAlgoImplBase
from ...algos.qlearning.torch.utility import (
    ContinuousQFunctionMixin,
    DiscreteQFunctionMixin,
)
from ...dataset import Shape
from ...models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
)
from ...torch_utility import Modules, TorchMiniBatch, hard_sync

__all__ = ["FQEBaseImpl", "FQEImpl", "DiscreteFQEImpl", "FQEBaseModules"]


@dataclasses.dataclass(frozen=True)
class FQEBaseModules(Modules):
    q_funcs: nn.ModuleList
    targ_q_funcs: nn.ModuleList
    optim: Optimizer


class FQEBaseImpl(QLearningAlgoImplBase):
    _algo: QLearningAlgoImplBase
    _modules: FQEBaseModules
    _gamma: float
    _q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _targ_q_func_forwarder: Union[
        DiscreteEnsembleQFunctionForwarder, ContinuousEnsembleQFunctionForwarder
    ]
    _target_update_interval: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        algo: QLearningAlgoImplBase,
        modules: FQEBaseModules,
        q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        targ_q_func_forwarder: Union[
            DiscreteEnsembleQFunctionForwarder,
            ContinuousEnsembleQFunctionForwarder,
        ],
        gamma: float,
        target_update_interval: int,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._algo = algo
        self._gamma = gamma
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._target_update_interval = target_update_interval
        hard_sync(modules.targ_q_funcs, modules.q_funcs)

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations, next_actions
            )

    def update_target(self) -> None:
        hard_sync(self._modules.targ_q_funcs, self._modules.q_funcs)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_update(
        self, batch: TorchMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        next_actions = self._algo.predict_best_action(batch.next_observations)

        q_tpn = self.compute_target(batch, next_actions)
        loss = self.compute_loss(batch, q_tpn)

        self._modules.optim.zero_grad()
        loss.backward()
        self._modules.optim.step()

        if grad_step % self._target_update_interval == 0:
            self.update_target()

        return {"loss": float(loss.cpu().detach().numpy())}


class FQEImpl(ContinuousQFunctionMixin, FQEBaseImpl):
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder


class DiscreteFQEImpl(DiscreteQFunctionMixin, FQEBaseImpl):
    _q_func_forwarder: ContinuousEnsembleQFunctionForwarder
    _targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                next_actions.long(),
            )
