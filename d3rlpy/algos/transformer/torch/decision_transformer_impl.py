import dataclasses
from typing import Dict

import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import ContinuousDecisionTransformer
from ....torch_utility import (
    Modules,
    TorchTrajectoryMiniBatch,
    eval_api,
    train_api,
)
from ..base import TransformerAlgoImplBase
from ..inputs import TorchTransformerInput

__all__ = ["DecisionTransformerImpl", "DecisionTransformerModules"]


@dataclasses.dataclass(frozen=True)
class DecisionTransformerModules(Modules):
    transformer: ContinuousDecisionTransformer
    optim: Optimizer


class DecisionTransformerImpl(TransformerAlgoImplBase):
    _modules: DecisionTransformerModules
    _scheduler: torch.optim.lr_scheduler.LambdaLR
    _clip_grad_norm: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DecisionTransformerModules,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        clip_grad_norm: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._scheduler = scheduler
        self._clip_grad_norm = clip_grad_norm

    @eval_api
    def predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # (1, T, A)
        action = self._modules.transformer(
            inpt.observations, inpt.actions, inpt.returns_to_go, inpt.timesteps
        )
        # (1, T, A) -> (A,)
        return action[0][-1]

    @train_api
    def update(self, batch: TorchTrajectoryMiniBatch) -> Dict[str, float]:
        self._modules.optim.zero_grad()

        loss = self.compute_loss(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._modules.transformer.parameters(), self._clip_grad_norm
        )
        self._modules.optim.step()
        self._scheduler.step()

        return {"loss": float(loss.cpu().detach().numpy())}

    def compute_loss(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        action = self._modules.transformer(
            batch.observations,
            batch.actions,
            batch.returns_to_go,
            batch.timesteps,
        )
        # (B, T, A) -> (B, T)
        loss = ((action - batch.actions) ** 2).sum(dim=-1)
        return (loss * batch.masks).sum() / batch.masks.sum()
