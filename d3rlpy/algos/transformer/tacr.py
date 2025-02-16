import dataclasses

from ...base import DeviceArg, register_learnable
from ...constants import ActionSpace, PositionEncodingType
from ...models import EncoderFactory, make_encoder_field
from ...models.builders import (
    create_continuous_decision_transformer,
    create_continuous_q_function,
)
from ...models.q_functions import QFunctionFactory, make_q_func_field
from ...optimizers import OptimizerFactory, make_optimizer_field
from ...types import Shape
from .base import TransformerAlgoBase, TransformerConfig
from .torch.tacr_impl import (
    TACRImpl,
    TACRModules,
)

__all__ = [
    "TACRConfig",
    "TACR",
]


@dataclasses.dataclass()
class TACRConfig(TransformerConfig):
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
    critic_learning_rate: float = 1e-4
    encoder_factory: EncoderFactory = make_encoder_field()
    critic_encoder_factory: EncoderFactory = make_encoder_field()
    optim_factory: OptimizerFactory = make_optimizer_field()
    critic_optim_factory: OptimizerFactory = make_optimizer_field()
    q_func_factory: QFunctionFactory = make_q_func_field()
    num_heads: int = 1
    num_layers: int = 3
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1
    activation_type: str = "relu"
    position_encoding_type: PositionEncodingType = PositionEncodingType.SIMPLE
    n_critics: int = 2
    alpha: float = 2.5
    tau: float = 0.005
    target_smoothing_sigma: float = 0.2
    target_smoothing_clip: float = 0.5
    compile_graph: bool = False

    def create(
        self, device: DeviceArg = False, enable_ddp: bool = False
    ) -> "TACR":
        return TACR(self, device, enable_ddp)

    @staticmethod
    def get_type() -> str:
        return "tacr"


class TACR(TransformerAlgoBase[TACRImpl, TACRConfig]):
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

        q_funcs, q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        targ_q_funcs, targ_q_func_forwarder = create_continuous_q_function(
            observation_shape,
            action_size,
            self._config.critic_encoder_factory,
            self._config.q_func_factory,
            n_ensembles=self._config.n_critics,
            device=self._device,
            enable_ddp=self._enable_ddp,
        )
        critic_optim = self._config.critic_optim_factory.create(
            q_funcs.named_modules(),
            lr=self._config.critic_learning_rate,
            compiled=self.compiled,
        )

        modules = TACRModules(
            transformer=transformer,
            optim=optim,
            q_funcs=q_funcs,
            targ_q_funcs=targ_q_funcs,
            critic_optim=critic_optim,
        )

        self._impl = TACRImpl(
            observation_shape=observation_shape,
            action_size=action_size,
            modules=modules,
            q_func_forwarder=q_func_forwarder,
            targ_q_func_forwarder=targ_q_func_forwarder,
            alpha=self._config.alpha,
            gamma=self._config.gamma,
            tau=self._config.tau,
            target_smoothing_sigma=self._config.target_smoothing_sigma,
            target_smoothing_clip=self._config.target_smoothing_clip,
            device=self._device,
            compiled=self.compiled,
        )

    def get_action_type(self) -> ActionSpace:
        return ActionSpace.CONTINUOUS


register_learnable(TACRConfig)
