import dataclasses
import math
from typing import Dict

import torch
import torch.nn.functional as F

from ....models import OptimizerWrapper
from ....models.torch import (
    ContinuousDecisionTransformer,
    DiscreteDecisionTransformer,
)
from ....torch_utility import Modules, TorchTrajectoryMiniBatch
from ....types import Shape
from ..base import TransformerAlgoImplBase
from ..inputs import TorchTransformerInput

__all__ = [
    "DecisionTransformerImpl",
    "DecisionTransformerModules",
    "DiscreteDecisionTransformerModules",
    "DiscreteDecisionTransformerImpl",
]


@dataclasses.dataclass(frozen=True)
class DecisionTransformerModules(Modules):
    transformer: ContinuousDecisionTransformer
    optim: OptimizerWrapper


class DecisionTransformerImpl(TransformerAlgoImplBase):
    _modules: DecisionTransformerModules
    _scheduler: torch.optim.lr_scheduler.LRScheduler

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DecisionTransformerModules,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._scheduler = scheduler

    def inner_predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # (1, T, A)
        action = self._modules.transformer(
            inpt.observations, inpt.actions, inpt.returns_to_go, inpt.timesteps
        )
        # (1, T, A) -> (A,)
        return action[0][-1]

    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self._modules.optim.step(grad_step)
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
        return loss.mean()


@dataclasses.dataclass(frozen=True)
class DiscreteDecisionTransformerModules(Modules):
    transformer: DiscreteDecisionTransformer
    optim: OptimizerWrapper


class DiscreteDecisionTransformerImpl(TransformerAlgoImplBase):
    _modules: DiscreteDecisionTransformerModules
    _warmup_tokens: int
    _final_tokens: int
    _initial_learning_rate: float
    _tokens: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteDecisionTransformerModules,
        warmup_tokens: int,
        final_tokens: int,
        initial_learning_rate: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=device,
        )
        self._warmup_tokens = warmup_tokens
        self._final_tokens = final_tokens
        self._initial_learning_rate = initial_learning_rate
        # TODO: Include stateful information in checkpoint.
        self._tokens = 0

    def inner_predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # (1, T, A)
        _, logits = self._modules.transformer(
            inpt.observations, inpt.actions, inpt.returns_to_go, inpt.timesteps
        )
        # (1, T, A) -> (A,)
        return logits[0][-1]

    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> Dict[str, float]:
        self._modules.optim.zero_grad()
        loss = self.compute_loss(batch)
        loss.backward()
        self._modules.optim.step(grad_step)

        # schedule learning rate
        self._tokens += int(batch.masks.sum().cpu().detach().numpy())
        if self._tokens < self._warmup_tokens:
            # linear warmup
            lr_mult = self._tokens / max(1, self._warmup_tokens)
        else:
            # cosine learning rate decay
            progress = (self._tokens - self._warmup_tokens) / max(
                1, self._final_tokens - self._warmup_tokens
            )
            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        new_learning_rate = lr_mult * self._initial_learning_rate
        for param_group in self._modules.optim.optim.param_groups:
            param_group["lr"] = new_learning_rate

        return {
            "loss": float(loss.cpu().detach().numpy()),
            "learning_rate": new_learning_rate,
        }

    def compute_loss(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        _, logits = self._modules.transformer(
            batch.observations,
            batch.actions,
            batch.returns_to_go,
            batch.timesteps,
        )
        loss = F.cross_entropy(
            logits.view(-1, self._action_size),
            batch.actions.view(-1).long(),
            reduction="none",
        )
        return loss.mean()
