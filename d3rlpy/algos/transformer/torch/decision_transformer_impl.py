import torch

from ....dataset import Shape
from ....models import (
    EncoderFactory,
    OptimizerFactory,
    create_continuous_decision_transformer,
    create_discrete_decision_transformer,
)
from ....models.torch import (
    ContinuousDecisionTransformer,
    DiscreteDecisionTransformer,
)
from ....torch_utility import (
    TorchTrajectoryMiniBatch,
    eval_api,
    to_device,
    train_api,
)
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
        learning_rate: float,
        encoder_factory: EncoderFactory,
        optim_factory: OptimizerFactory,
        num_heads: int,
        max_timestep: int,
        num_layers: int,
        context_size: int,
        attn_dropout: float,
        resid_dropout: float,
        embed_dropout: float,
        activation_type: str,
        position_encoding_type: str,
        warmup_steps: int,
        clip_grad_norm: float,
        device: str,
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            device=device,
        )
        self._transformer = create_continuous_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=encoder_factory,
            num_heads=num_heads,
            max_timestep=max_timestep,
            num_layers=num_layers,
            context_size=context_size,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            activation_type=activation_type,
            position_encoding_type=position_encoding_type,
        )

        to_device(self, device)

        self._optim = optim_factory.create(
            self._transformer.parameters(), lr=learning_rate
        )
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._optim, lambda steps: min((steps + 1) / warmup_steps, 1)
        )
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
        return (loss * batch.masks).mean()
