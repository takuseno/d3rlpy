import dataclasses
import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch.optim import Optimizer

from ....models.torch import (
    ContinuousDecisionTransformer,
    DiscreteDecisionTransformer,
)
from ....torch_utility import Modules, TorchTrajectoryMiniBatch, eval_api
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
    optim: Optimizer


class DecisionTransformerImpl(TransformerAlgoImplBase):
    _modules: DecisionTransformerModules
    _scheduler: torch.optim.lr_scheduler.LRScheduler
    _clip_grad_norm: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DecisionTransformerModules,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
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

    def inner_update(
        self, batch: TorchTrajectoryMiniBatch, grad_step: int
    ) -> Dict[str, float]:
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
        return loss.mean()


@dataclasses.dataclass(frozen=True)
class DiscreteDecisionTransformerModules(Modules):
    transformer: DiscreteDecisionTransformer
    optim: Optimizer


class DiscreteDecisionTransformerImpl(TransformerAlgoImplBase):
    _modules: DiscreteDecisionTransformerModules
    _clip_grad_norm: float
    _warmup_tokens: int
    _final_tokens: int
    _initial_learning_rate: float
    _tokens: int

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        modules: DiscreteDecisionTransformerModules,
        clip_grad_norm: float,
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
        self._clip_grad_norm = clip_grad_norm
        self._warmup_tokens = warmup_tokens
        self._final_tokens = final_tokens
        self._initial_learning_rate = initial_learning_rate
        # TODO: Include stateful information in checkpoint.
        self._tokens = 0

    @eval_api
    def predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
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
        torch.nn.utils.clip_grad_norm_(
            self._modules.transformer.parameters(), self._clip_grad_norm
        )
        self._modules.optim.step()

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
        for param_group in self._modules.optim.param_groups:
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
