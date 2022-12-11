from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Type

from dataclasses_json import config

from .torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
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
    "Q_FUNC_LIST",
    "make_q_func_field",
]


@dataclass(frozen=True)
class QFunctionFactory:
    share_encoder: bool = False

    def create_discrete(
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: an encoder module that processes the observation to
                obtain feature representations.
            action_size: dimension of discrete action-space.

        Returns:
            discrete Q function object.

        """
        raise NotImplementedError

    def create_continuous(
        self, encoder: EncoderWithAction
    ) -> ContinuousQFunction:
        """Returns PyTorch's Q function module.

        Args:
            encoder: an encoder module that processes the observation and
                action to obtain feature representations.

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


@dataclass(frozen=True)
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
        action_size: int,
    ) -> DiscreteMeanQFunction:
        return DiscreteMeanQFunction(encoder, action_size)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousMeanQFunction:
        return ContinuousMeanQFunction(encoder)

    @staticmethod
    def get_type() -> str:
        return "mean"


@dataclass(frozen=True)
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
        self, encoder: Encoder, action_size: int
    ) -> DiscreteQRQFunction:
        return DiscreteQRQFunction(encoder, action_size, self.n_quantiles)

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousQRQFunction:
        return ContinuousQRQFunction(encoder, self.n_quantiles)

    @staticmethod
    def get_type() -> str:
        return "qr"


@dataclass(frozen=True)
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
        action_size: int,
    ) -> DiscreteIQNQFunction:
        return DiscreteIQNQFunction(
            encoder=encoder,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousIQNQFunction:
        return ContinuousIQNQFunction(
            encoder=encoder,
            n_quantiles=self.n_quantiles,
            n_greedy_quantiles=self.n_greedy_quantiles,
            embed_size=self.embed_size,
        )

    @staticmethod
    def get_type() -> str:
        return "iqn"


@dataclass(frozen=True)
class FQFQFunctionFactory(QFunctionFactory):
    """Fully parameterized Quantile Function Q function factory.

    References:
        * `Yang et al., Fully parameterized quantile function for
          distributional reinforcement learning.
          <https://arxiv.org/abs/1911.02140>`_

    Args:
        share_encoder (bool): flag to share encoder over multiple Q functions.
        n_quantiles: the number of quantiles.
        embed_size: the embedding size.
        entropy_coeff: the coefficiency of entropy penalty term.

    """

    n_quantiles: int = 32
    embed_size: int = 64
    entropy_coeff: float = 0.0

    def create_discrete(
        self,
        encoder: Encoder,
        action_size: int,
    ) -> DiscreteFQFQFunction:
        return DiscreteFQFQFunction(
            encoder=encoder,
            action_size=action_size,
            n_quantiles=self.n_quantiles,
            embed_size=self.embed_size,
            entropy_coeff=self.entropy_coeff,
        )

    def create_continuous(
        self,
        encoder: EncoderWithAction,
    ) -> ContinuousFQFQFunction:
        return ContinuousFQFQFunction(
            encoder=encoder,
            n_quantiles=self.n_quantiles,
            embed_size=self.embed_size,
            entropy_coeff=self.entropy_coeff,
        )

    @staticmethod
    def get_type() -> str:
        return "fqf"


Q_FUNC_LIST: Dict[str, Type[QFunctionFactory]] = {}


def register_q_func_factory(cls: Type[QFunctionFactory]) -> None:
    """Registers Q function factory class.

    Args:
        cls: Q function factory class inheriting ``QFunctionFactory``.

    """
    type_name = cls.get_type()
    is_registered = type_name in Q_FUNC_LIST
    assert not is_registered, f"{type_name} seems to be already registered"
    Q_FUNC_LIST[type_name] = cls


def create_q_func_factory(name: str, **kwargs: Any) -> QFunctionFactory:
    """Returns registered Q function factory object.

    Args:
        name: registered Q function factory type name.
        kwargs: Q function arguments.

    Returns:
        Q function factory object.

    """
    assert name in Q_FUNC_LIST, f"{name} seems not to be registered."
    factory = Q_FUNC_LIST[name](**kwargs)
    assert isinstance(factory, QFunctionFactory)
    return factory


def _encoder(q_func: QFunctionFactory) -> Dict[str, Any]:
    return {"type": q_func.get_type(), "params": asdict(q_func)}


def _decoder(dict_config: Dict[str, Any]) -> QFunctionFactory:
    return create_q_func_factory(dict_config["type"], **dict_config["params"])


def make_q_func_field() -> QFunctionFactory:
    return field(
        metadata=config(encoder=_encoder, decoder=_decoder),
        default=MeanQFunctionFactory(),
    )


register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
register_q_func_factory(FQFQFunctionFactory)
