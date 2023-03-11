import copy

import torch
from torch.optim import Optimizer

from ...algos.qlearning.base import QLearningAlgoImplBase
from ...algos.qlearning.torch.utility import (
    ContinuousQFunctionMixin,
    DiscreteQFunctionMixin,
)
from ...dataset import Shape
from ...models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
    EnsembleQFunction,
)
from ...torch_utility import TorchMiniBatch, hard_sync, train_api

__all__ = ["FQEBaseImpl", "FQEImpl", "DiscreteFQEImpl"]


class FQEBaseImpl(QLearningAlgoImplBase):
    _gamma: float
    _q_func: EnsembleQFunction
    _targ_q_func: EnsembleQFunction
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleQFunction,
        optim: Optimizer,
        gamma: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._gamma = gamma
        self._q_func = q_func
        self._targ_q_func = copy.deepcopy(q_func)
        self._optim = optim

    @train_api
    def update(
        self, batch: TorchMiniBatch, next_actions: torch.Tensor
    ) -> float:
        q_tpn = self.compute_target(batch, next_actions)
        loss = self.compute_loss(batch, q_tpn)

        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

        return float(loss.cpu().detach().numpy())

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func.compute_error(
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
            return self._targ_q_func.compute_target(
                batch.next_observations, next_actions
            )

    def update_target(self) -> None:
        hard_sync(self._targ_q_func, self._q_func)

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class FQEImpl(ContinuousQFunctionMixin, FQEBaseImpl):
    _q_func: EnsembleContinuousQFunction
    _targ_q_func: EnsembleContinuousQFunction


class DiscreteFQEImpl(DiscreteQFunctionMixin, FQEBaseImpl):
    _q_func: EnsembleDiscreteQFunction
    _targ_q_func: EnsembleDiscreteQFunction

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        return self._q_func.compute_error(
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
            return self._targ_q_func.compute_target(
                batch.next_observations,
                next_actions.long(),
            )
