# pylint: disable=too-many-ancestors

import torch
from torch import nn
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
)
from ....torch_utility import Checkpointer, TorchMiniBatch
from .td3_impl import TD3Impl

__all__ = ["TD3PlusBCImpl"]


class TD3PlusBCImpl(TD3Impl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        policy: DeterministicPolicy,
        q_funcs: nn.ModuleList,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_funcs: nn.ModuleList,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        checkpointer: Checkpointer,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            policy=policy,
            q_funcs=q_funcs,
            q_func_forwarder=q_func_forwarder,
            targ_q_funcs=targ_q_funcs,
            targ_q_func_forwarder=targ_q_func_forwarder,
            actor_optim=actor_optim,
            critic_optim=critic_optim,
            gamma=gamma,
            tau=tau,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            checkpointer=checkpointer,
            device=device,
        )
        self._alpha = alpha

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        action = self._policy(batch.observations).squashed_mu
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        return lam * -q_t.mean() + ((batch.actions - action) ** 2).mean()
