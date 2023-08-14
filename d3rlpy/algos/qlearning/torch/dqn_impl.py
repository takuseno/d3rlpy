from typing import Dict

import torch
from torch import nn
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import DiscreteEnsembleQFunctionForwarder
from ....torch_utility import Checkpointer, TorchMiniBatch, hard_sync, train_api
from ..base import QLearningAlgoImplBase
from .utility import DiscreteQFunctionMixin

__all__ = ["DQNImpl", "DoubleDQNImpl"]


class DQNImpl(DiscreteQFunctionMixin, QLearningAlgoImplBase):
    _gamma: float
    _q_funcs: nn.ModuleList
    _targ_q_funcs: nn.ModuleList
    _q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder
    _optim: Optimizer

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        q_funcs: nn.ModuleList,
        q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: DiscreteEnsembleQFunctionForwarder,
        optim: Optimizer,
        gamma: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            checkpointer=checkpointer,
            device=device,
        )
        self._gamma = gamma
        self._q_funcs = q_funcs
        self._q_func_forwarder = q_func_forwarder
        self._targ_q_funcs = targ_q_funcs
        self._targ_q_func_forwarder = targ_q_func_forwarder
        self._optim = optim
        hard_sync(targ_q_funcs, q_funcs)

    @train_api
    def update(self, batch: TorchMiniBatch) -> Dict[str, float]:
        self._optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_loss(batch, q_tpn)

        loss.backward()
        self._optim.step()

        return {"loss": float(loss.cpu().detach().numpy())}

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

    def inner_predict_best_action(self, x: torch.Tensor) -> torch.Tensor:
        return self._q_func_forwarder.compute_expected_q(x).argmax(dim=1)

    def inner_sample_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner_predict_best_action(x)

    def update_target(self) -> None:
        hard_sync(self._targ_q_funcs, self._q_funcs)

    @property
    def q_function(self) -> nn.ModuleList:
        return self._q_funcs

    @property
    def q_function_optim(self) -> Optimizer:
        return self._optim


class DoubleDQNImpl(DQNImpl):
    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self.inner_predict_best_action(batch.next_observations)
            return self._targ_q_func_forwarder.compute_target(
                batch.next_observations,
                action,
                reduction="min",
            )
