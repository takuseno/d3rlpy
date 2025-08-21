import dataclasses
from typing import Optional

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace, PositionEncodingType
from ...models import EncoderFactory, make_encoder_field
from ...models.builders import (
    create_continuous_decision_transformer,
    create_discrete_decision_transformer,
)
from ...optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
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
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE
    embedding_size: Optional[int] = None
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DecisionTransformer":
        return DecisionTransformer(self, device, enable_ddp)

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
            enable_ddp=self._enable_ddp,
        )
        optim = self._config.optim_factory.create(
            transformer.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            device=self._device,
            compiled=self.compiled,
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
        max_timestep (int): Maximum environmental timestep.
        batch_size (int): Mini-batch size.
        learning_rate (float): Learning rate.
        encoder_factory (d3rlpy.models.encoders.EncoderFactory):
            Encoder factory.
        optim_factory (d3rlpy.optimizers.OptimizerFactory):
            Optimizer factory.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of attention blocks.
        attn_dropout (float): Dropout probability for attentions.
        resid_dropout (float): Dropout probability for residual connection.
        embed_dropout (float): Dropout probability for embeddings.
        activation_type (str): Type of activation function.
        embed_activation_type (str): Type of activation function applied to
            embeddings.
        position_encoding_type (d3rlpy.PositionEncodingType):
            Type of positional encoding (``SIMPLE`` or ``GLOBAL``).
        warmup_tokens (int): Number of tokens to warmup learning rate scheduler.
        final_tokens (int): Final number of tokens for learning rate scheduler.
        compile_graph (bool): Flag to enable JIT compilation and CUDAGraph.
    """

    batch_size: int = 128
    learning_rate: float = 6e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    num_heads: int = 8
    num_layers: int = 6
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "gelu"
    embed_activation_type: str = "tanh"
    position_encoding_type: PositionEncodingType = PositionEncodingType.GLOBAL
    warmup_tokens: int = 10240
    final_tokens: int = 30000000
    embedding_size: Optional[int] = None
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "DiscreteDecisionTransformer":
        return DiscreteDecisionTransformer(self, device, enable_ddp)

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
            enable_ddp=self._enable_ddp,
            embedding_size=self._config.embedding_size,
        )
        optim = self._config.optim_factory.create(
            transformer.named_modules(),
            lr=self._config.learning_rate,
            compiled=self.compiled,
        )

        modules = DiscreteDecisionTransformerModules(
            transformer=transformer,
            optim=optim,
        )

        self._impl = DiscreteDecisionTransformerImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            warmup_tokens=self._config.warmup_tokens,
            final_tokens=self._config.final_tokens,
            initial_learning_rate=self._config.learning_rate,
            compiled=self.compiled,
            device=self._device,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.DISCRETE


register_learnable(DecisionTransformerConfig)
register_learnable(DiscreteDecisionTransformerConfig)
