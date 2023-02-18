import dataclasses
from typing import Dict

import torch

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace
from ...dataset import Shape
from ...models import (
    EncoderFactory,
    OptimizerFactory,
    make_encoder_field,
    make_optimizer_field,
)
from ...models.builders import create_continuous_decision_transformer
from ...torch_utility import TorchTrajectoryMiniBatch
from .base import TransformerAlgoBase, TransformerConfig
from .torch.decision_transformer_impl import DecisionTransformerImpl

__all__ = ["DecisionTransformerConfig", "DecisionTransformer"]


@dataclasses.dataclass()
class DecisionTransformerConfig(TransformerConfig):
    batch_size: int = 64
    learning_rate: float = 1e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 1
    max_timestep: int = 1000
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: str = "simple"
    warmup_steps: int = 10000
    clip_grad_norm: float = 0.25

    def create(self, device: DeviceArg = False) -> "DecisionTransformer":
        return DecisionTransformer(self, device)

    @staticmethod
    def get_type() -> str:
        return "decision_transformer"


class DecisionTransformer(TransformerAlgoBase):
    _config: DecisionTransformerConfig
    _impl: DecisionTransformerImpl

    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_continuous_decision_transformer(
            observation_shape=observation_shape,
            action_size=action_size,
            encoder_factory=self._config.encoder_factory,
            num_heads=self._config.num_heads,
            max_timestep=self._config.max_timestep,
            num_layers=self._config.num_layers,
            context_size=self._config.context_size,
            attn_dropout=self._config.attn_dropout,
            resid_dropout=self._config.resid_dropout,
            embed_dropout=self._config.embed_dropout,
            activation_type=self._config.activation_type,
            position_encoding_type=self._config.position_encoding_type,
            device=self._device,
        )
        optim = self._config.optim_factory.create(
            transformer.parameters(), lr=self._config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda steps: min((steps + 1) / self._config.warmup_steps, 1)
        )

        self._impl = DecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            transformer=transformer,
            optim=optim,
            scheduler=scheduler,
            clip_grad_norm=self._config.clip_grad_norm,
            device=self._device,
        )

    def inner_update(self, batch: TorchTrajectoryMiniBatch) -> Dict[str, float]:
        assert self._impl
        loss = self._impl.update(batch)
        return {"loss": loss}

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(DecisionTransformerConfig)
