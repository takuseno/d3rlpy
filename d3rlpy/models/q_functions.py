import dataclasses

from ..serializable_config import DynamicConfig, generate_config_registration
from .torch import (
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQFunction,
    ContinuousQRQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQFunction,
    DiscreteQRQFunction,
    Encoder,
    EncoderWithAction,
)

__all__ = [
    "QFunctionFactory",
    "MeanQFunctionFactory",
    "QRQFunctionFactory",
    "IQNQFunctionFactory",
    "make_q_func_field",
]


@dataclasses.dataclass()
class QFunctionFactory(DynamicConfig):
    share_encoder: bool = False

    def create_discrete(
        self, encoder: Encoder, hidden_size: int, action_size: int
    ) -> DiscreteQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: Encoder that processes the observation to
                obtain feature representations.
            hidden_size: Dimension of encoder output.
            action_size: Dimension of discrete action-space.

        Returns:
            discrete Q function object.
        """
        raise NotImplementedError

    def create_continuous(
        self, encoder: EncoderWithAction, hidden_size: int, action_size: int
    ) -> ContinuousQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: Encoder module that processes the observation and
                action to obtain feature representations.
            hidden_size: Dimension of encoder output.
            action_size: Dimension of continuous actions.

        Returns:
            continuous Q function object.
        """
        raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns Q function type.

        Returns:
            Q function type.
        """
        raise NotImplementedError


@dataclasses.dataclass()
class MeanQFunctionFactory(QFunctionFactory):
    """Standard Q function factory class.

    This is the standard Q function factory class.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
    """

    def create_discrete(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
    ) -> DiscreteMeanQFunction:
        return DiscreteMeanQFunction(encoder, hidden_size, action_size)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        action_size: int,
    ) -> ContinuousMeanQFunction:
        return ContinuousMeanQFunction(encoder, hidden_size, action_size)

    @staticmethod
    def get_type() -> str:
        return "mean"


@dataclasses.dataclass()
class QRQFunctionFactory(QFunctionFactory):
    """Quantile Regression Q function factory class.

    References:
        * `Dabney et al., Distributional reinforcement learning with quantile
          regression. <https://arxiv.org/abs/1710.10044>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
    """

    n_quantiles: int = 32

    def create_discrete(
        self, encoder: Encoder, hidden_size: int, action_size: int
    ) -> DiscreteQRQFunction:
        return DiscreteQRQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
        action_size: int,
    ) -> ContinuousQRQFunction:
        return ContinuousQRQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
        )

    @staticmethod
    def get_type() -> str:
        return "qr"


@dataclasses.dataclass()
class IQNQFunctionFactory(QFunctionFactory):
    """Implicit Quantile Network Q function factory class.

    References:
        * `Dabney et al., Implicit quantile networks for distributional
          reinforcement learning. <https://arxiv.org/abs/1806.06923>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        n_greedy_quantiles: the number of quantiles for inference.
        embed_size: the embedding size.
    """

    n_quantiles: int = 64
    n_greedy_quantiles: int = 32
    embed_size: int = 64

    def create_discrete(
        self,
        encoder: Encoder,
        hidden_size: int,
        action_size: int,
    ) -> DiscreteIQNQFunction:
        return DiscreteIQNQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )

    def create_continuous(
        self, encoder: EncoderWithAction, hidden_size: int, action_size: int
    ) -> ContinuousIQNQFunction:
        return ContinuousIQNQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )

    @staticmethod
    def get_type() -> str:
        return "iqn"


register_q_func_factory, make_q_func_field = generate_config_registration(
    QFunctionFactory, lambda: MeanQFunctionFactory()
)


register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
