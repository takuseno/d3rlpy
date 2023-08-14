# pylint: disable=too-many-ancestors

import torch

from ....dataset import Shape
from ....models.torch import ContinuousEnsembleQFunctionForwarder
from ....torch_utility import TorchMiniBatch
from .ddpg_impl import DDPGModules
from .td3_impl import TD3Impl

__all__ = ["TD3PlusBCImpl"]


class TD3PlusBCImpl(TD3Impl):
    _alpha: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DDPGModules,
        q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        targ_q_func_forwarder: ContinuousEnsembleQFunctionForwarder,
        gamma: float,
        tau: float,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        alpha: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            gamma=gamma,
            tau=tau,
            target_smoothing_sigma=target_smoothing_sigma,
            target_smoothing_clip=target_smoothing_clip,
            device=device,
        )
        self._alpha = alpha

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        action = self._modules.policy(batch.observations).squashed_mu
        q_t = self._q_func_forwarder.compute_expected_q(
            batch.observations, action, "none"
        )[0]
        lam = self._alpha / (q_t.abs().mean()).detach()
        return lam * -q_t.mean() + ((batch.actions - action) ** 2).mean()
