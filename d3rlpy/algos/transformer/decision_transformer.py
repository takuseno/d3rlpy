import dataclasses

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
from ...models.builders import (
    create_continuous_decision_transformer,
    create_discrete_decision_transformer,
)
from .base import TransformerAlgoBase, TransformerConfig
from .torch.decision_transformer_impl import (
    DecisionTransformerImpl,
    DecisionTransformerModules,
    DiscreteDecisionTransformerImpl,
    DiscreteDecisionTransformerModules,
)

__all__ = [
    "DecisionTransformerConfig",
    "DecisionTransformer",
    "DiscreteDecisionTransformerConfig",
    "DiscreteDecisionTransformer",
]


@dataclasses.dataclass()
class DecisionTransformerConfig(TransformerConfig):
    """Config of Decision Transformer.

    Decision Transformer solves decision-making problems as a sequence modeling
    problem.

    References:
        * `Chen at el., Decision Transformer: Reinforcement Learning via
          Sequence Modeling. <https://arxiv.org/abs/2106.01345>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        action_scaler (d3rlpy.preprocessing.ActionScaler): Action preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        max_timestep (int): Maximum environmental timestep.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        position_encoding_type (str): Type of positional encoding
            (``simple`` or ``global``).
        warmup_steps (int): Warmup steps for learning rate scheduler.
        clip_grad_norm (float): Norm of gradient clipping.
        compile (bool): (experimental) Flag to enable JIT compilation.
    """

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
    compile: bool = False

    def create(self, device: DeviceArg = False) -> "DecisionTransformer":
        return DecisionTransformer(self, device)

    @staticmethod
    def get_type() -> str:
        return "decision_transformer"


class DecisionTransformer(
    TransformerAlgoBase[DecisionTransformerImpl, DecisionTransformerConfig]
):
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
            transformer.named_modules(), lr=self._config.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda steps: min((steps + 1) / self._config.warmup_steps, 1)
        )

        # JIT compile
        if self._config.compile:
            transformer = torch.compile(transformer, fullgraph=True)

        modules = DecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            scheduler=scheduler,
            clip_grad_norm=self._config.clip_grad_norm,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


@dataclasses.dataclass()
class DiscreteDecisionTransformerConfig(TransformerConfig):
    """Config of Decision Transformer for discrte action-space.

    Decision Transformer solves decision-making problems as a sequence modeling
    problem.

    References:
        * `Chen at el., Decision Transformer: Reinforcement Learning via
          Sequence Modeling. <https://arxiv.org/abs/2106.01345>`_

    Args:
        observation_scaler (d3rlpy.preprocessing.ObservationScaler):
            Observation preprocessor.
        reward_scaler (d3rlpy.preprocessing.RewardScaler): Reward preprocessor.
        context_size (int): Prior sequence length.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.models.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        max_timestep (int): Maximum environmental timestep.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        embed_activation_type (str): Type of activation function applied to
            embeddings.
        position_encoding_type (str): Type of positional encoding
            (``simple`` or ``global``).
        warmup_tokens (int): Number of tokens to warmup learning rate scheduler.
        final_tokens (int): Final number of tokens for learning rate scheduler.
        clip_grad_norm (float): Norm of gradient clipping.
        compile (bool): (experimental) Flag to enable JIT compilation.
    """

    batch_size: int = 128
    learning_rate: float = 6e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 8
    max_timestep: int = 1000
    num_layers: int = 6
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "gelu"
    embed_activation_type: str = "tanh"
    position_encoding_type: str = "global"
    warmup_tokens: int = 10240
    final_tokens: int = 30000000
    clip_grad_norm: float = 1.0
    compile: bool = False

    def create(
        self, device: DeviceArg = False
    ) -> "DiscreteDecisionTransformer":
        return DiscreteDecisionTransformer(self, device)

    @staticmethod
    def get_type() -> str:
        return "discrete_decision_transformer"


class DiscreteDecisionTransformer(
    TransformerAlgoBase[
        DiscreteDecisionTransformerImpl, DiscreteDecisionTransformerConfig
    ]
):
    def inner_create_impl(
        self, observation_shape: Shape, action_size: int
    ) -> None:
        transformer = create_discrete_decision_transformer(
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
            embed_activation_type=self._config.embed_activation_type,
            position_encoding_type=self._config.position_encoding_type,
            device=self._device,
        )
        optim = self._config.optim_factory.create(
            transformer.named_modules(), lr=self._config.learning_rate
        )
        # JIT compile
        if self._config.compile:
            transformer = torch.compile(transformer, fullgraph=True)

        modules = DiscreteDecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DiscreteDecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            clip_grad_norm=self._config.clip_grad_norm,
            warmup_tokens=self._config.warmup_tokens,
            final_tokens=self._config.final_tokens,
            initial_learning_rate=self._config.learning_rate,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(DecisionTransformerConfig)
register_learnable(DiscreteDecisionTransformerConfig)
