import torch
from torch.optim import Optimizer

from ....dataset import Shape
from ....models.torch import ContinuousDecisionTransformer
from ....torch_utility import TorchTrajectoryMiniBatch, eval_api, train_api
from ..base import TransformerAlgoImplBase
from ..inputs import TorchTransformerInput

__all__ = ["DecisionTransformerImpl"]


class DecisionTransformerImpl(TransformerAlgoImplBase):
    _transformer: ContinuousDecisionTransformer
    _optim: torch.optim.Optimizer
    _scheduler: torch.optim.lr_scheduler.LambdaLR
    _clip_grad_norm: float

    def __init__(
        self,
        observation_shape: Shape,
        action_size: int,
        transformer: ContinuousDecisionTransformer,
        optim: Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        clip_grad_norm: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._transformer = transformer
        self._optim = optim
        self._scheduler = scheduler
        self._clip_grad_norm = clip_grad_norm

    @eval_api
    def predict(self, inpt: TorchTransformerInput) -> torch.Tensor:
        # (1, T, A)
        action = self._transformer(
            inpt.observations, inpt.actions, inpt.returns_to_go, inpt.timesteps
        )
        # (1, T, A) -> (A,)
        return action[0][-1]

    @train_api
    def update(self, batch: TorchTrajectoryMiniBatch) -> float:
        self._optim.zero_grad()

        loss = self.compute_loss(batch)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._transformer.parameters(), self._clip_grad_norm
        )
        self._optim.step()
        self._scheduler.step()

        return float(loss.cpu().detach().numpy())

    def compute_loss(self, batch: TorchTrajectoryMiniBatch) -> torch.Tensor:
        action = self._transformer(
            batch.observations,
            batch.actions,
            batch.returns_to_go,
            batch.timesteps,
        )
        # (B, T, A) -> (B, T)
        loss = ((action - batch.actions) ** 2).sum(dim=-1)
        return (loss * batch.masks).sum() / batch.masks.sum()
