import dataclasses

from ..serializable_config import DynamicConfig, generate_config_registration
from .torch import (
    ContinuousIQNQFunction,
    ContinuousIQNQFunctionForwarder,
    ContinuousMeanQFunction,
    ContinuousMeanQFunctionForwarder,
    ContinuousQFunction,
    ContinuousQFunctionForwarder,
    ContinuousQRQFunction,
    ContinuousQRQFunctionForwarder,
    DiscreteIQNQFunction,
    DiscreteIQNQFunctionForwarder,
    DiscreteMeanQFunction,
    DiscreteMeanQFunctionForwarder,
    DiscreteQFunction,
    DiscreteQFunctionForwarder,
    DiscreteQRQFunction,
    DiscreteQRQFunctionForwarder,
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
    ) -> tuple[DiscreteQFunction, DiscreteQFunctionForwarder]:
        """Returns PyTorch's Q function module.

        Args:
            encoder: Encoder that processes the observation to
                obtain feature representations.
            hidden_size: Dimension of encoder output.
            action_size: Dimension of discrete action-space.

        Returns:
            Tuple of discrete Q function and its forwarder.
        """
        raise NotImplementedError

    def create_continuous(
        self, encoder: EncoderWithAction, hidden_size: int
    ) -> tuple[ContinuousQFunction, ContinuousQFunctionForwarder]:
        """Returns PyTorch's Q function module.

        Args:
            encoder: Encoder module that processes the observation and
                action to obtain feature representations.
            hidden_size: Dimension of encoder output.

        Returns:
            Tuple of continuous Q function and its forwarder.
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
    ) -> tuple[DiscreteMeanQFunction, DiscreteMeanQFunctionForwarder]:
        q_func = DiscreteMeanQFunction(encoder, hidden_size, action_size)
        forwarder = DiscreteMeanQFunctionForwarder(q_func, action_size)
        return q_func, forwarder

    def create_continuous(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
    ) -> tuple[ContinuousMeanQFunction, ContinuousMeanQFunctionForwarder]:
        q_func = ContinuousMeanQFunction(encoder, hidden_size)
        forwarder = ContinuousMeanQFunctionForwarder(q_func)
        return q_func, forwarder

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
    ) -> tuple[DiscreteQRQFunction, DiscreteQRQFunctionForwarder]:
        q_func = DiscreteQRQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
        )
        forwarder = DiscreteQRQFunctionForwarder(q_func, self.n_quantiles)
        return q_func, forwarder

    def create_continuous(
        self,
        encoder: EncoderWithAction,
        hidden_size: int,
    ) -> tuple[ContinuousQRQFunction, ContinuousQRQFunctionForwarder]:
        q_func = ContinuousQRQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            n_quantiles=self.n_quantiles,
        )
        forwarder = ContinuousQRQFunctionForwarder(q_func, self.n_quantiles)
        return q_func, forwarder

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
    ) -> tuple[DiscreteIQNQFunction, DiscreteIQNQFunctionForwarder]:
        q_func = DiscreteIQNQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )
        forwarder = DiscreteIQNQFunctionForwarder(q_func, self.n_quantiles)
        return q_func, forwarder

    def create_continuous(
        self, encoder: EncoderWithAction, hidden_size: int
    ) -> tuple[ContinuousIQNQFunction, ContinuousIQNQFunctionForwarder]:
        q_func = ContinuousIQNQFunction(
            encoder=encoder,
            hidden_size=hidden_size,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )
        forwarder = ContinuousIQNQFunctionForwarder(
            q_func, self.n_greedy_quantiles
        )
        return q_func, forwarder

    @staticmethod
    def get_type() -> str:
        return "iqn"


register_q_func_factory, make_q_func_field = generate_config_registration(
    QFunctionFactory, lambda: MeanQFunctionFactory()
)


register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
