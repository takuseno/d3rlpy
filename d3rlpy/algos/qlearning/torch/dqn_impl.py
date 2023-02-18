import copy

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import EnsembleDiscreteQFunction, EnsembleQFunction
from ....torch_utility import TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase
from .utility import DiscreteQFunctionMixin

__all__ = ["DQNImpl", "DoubleDQNImpl"]


class DQNImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):

    _gamma: float
    _q_func: EnsembleDiscreteQFunction
    _targ_q_func: EnsembleDiscreteQFunction
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_func: EnsembleDiscreteQFunction,
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
        self._optim = optim
        self._targ_q_func = copy.deepcopy(q_func)

    @train_api
    def update(self, batch: TorchMiniBatch) -> float:
        assert self._optim is not None

        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_loss(batch, q_tpn)

        loss.backward()
        self._optim.step()

        return float(loss.cpu().detach().numpy())

    def compute_loss(
        self,
        batch: TorchMiniBatch,
        q_tpn: torch.Tensor,
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions.long(),
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        )

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            next_actions = self._targ_q_func(batch.next_observations)
            max_action = next_actions.argmax(dim=1)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                max_action,
                reduction="min",
            )

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func(x).argmax(dim=1)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        hard_sync(self._targ_q_func, self._q_func)

    @property
    def q_function(self) -> EnsembleQFunction:
        assert self._q_func
        return self._q_func

    @property
    def q_function_optim(self) -> Optimizer:
        assert self._optim
        return self._optim


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
