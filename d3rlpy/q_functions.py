from abc import ABCMeta, abstractmethod
from typing import Any, ClassVar, Dict, Optional, Union, Type
from .models.torch import Encoder, EncoderWithAction
from .models.torch import DiscreteQFunction
from .models.torch import ContinuousQFunction
from .models.torch import DiscreteMeanQFunction
from .models.torch import DiscreteQRQFunction
from .models.torch import DiscreteIQNQFunction
from .models.torch import DiscreteFQFQFunction
from .models.torch import ContinuousMeanQFunction
from .models.torch import ContinuousQRQFunction
from .models.torch import ContinuousIQNQFunction
from .models.torch import ContinuousFQFQFunction


class QFunctionFactory(metaclass=ABCMeta):
    TYPE: ClassVar[str] = "none"

    @abstractmethod
    def create(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        action_size: Optional[int] = None,
    ) -> Union[DiscreteQFunction, ContinuousQFunction]:
        """Returns PyTorch's Q function module.

        Args:
            encoder (torch.nn.Module): an encoder module that processes
                the observation (and action in continuous action-space) to
                obtain feature representations.
            action_size (int): dimension of discrete action-space. If the
                action-space is continous, ``None`` will be passed.

        Returns:
            torch.nn.Module: Q function object.

        """
        pass

    def get_type(self) -> str:
        """Returns Q function type.

        Returns:
            str: Q function type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns Q function parameters.

        Returns:
            dict: Q function parameters.

        """
        pass


class MeanQFunctionFactory(QFunctionFactory):
    """Standard Q function factory class.

    This is the standard Q function factory class.

    References:
        * `Mnih et al., Human-level control through deep reinforcement
          learning. <https://www.nature.com/articles/nature14236>`_
        * `Lillicrap et al., Continuous control with deep reinforcement
          learning. <https://arxiv.org/abs/1509.02971>`_

    """

    TYPE: ClassVar[str] = "mean"

    def create(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        action_size: Optional[int] = None,
    ) -> Union[ContinuousMeanQFunction, DiscreteMeanQFunction]:
        q_func: Union[ContinuousMeanQFunction, DiscreteMeanQFunction]
        if action_size is None:
            assert isinstance(encoder, EncoderWithAction)
            q_func = ContinuousMeanQFunction(encoder)
        else:
            assert isinstance(encoder, Encoder)
            q_func = DiscreteMeanQFunction(encoder, action_size)
        return q_func

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {}


class QRQFunctionFactory(QFunctionFactory):
    """Quantile Regression Q function factory class.

    References:
        * `Dabney et al., Distributional reinforcement learning with quantile
          regression. <https://arxiv.org/abs/1710.10044>`_

    Args:
        n_quantiles (int): the number of quantiles.

    """

    TYPE: ClassVar[str] = "qr"
    _n_quantiles: int

    def __init__(self, n_quantiles: int = 200):
        self._n_quantiles = n_quantiles

    def create(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        action_size: Optional[int] = None,
    ) -> Union[ContinuousQRQFunction, DiscreteQRQFunction]:
        q_func: Union[ContinuousQRQFunction, DiscreteQRQFunction]
        if action_size is None:
            assert isinstance(encoder, EncoderWithAction)
            q_func = ContinuousQRQFunction(encoder, self._n_quantiles)
        else:
            assert isinstance(encoder, Encoder)
            q_func = DiscreteQRQFunction(
                encoder, action_size, self._n_quantiles
            )
        return q_func

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {"n_quantiles": self._n_quantiles}

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles


class IQNQFunctionFactory(QFunctionFactory):
    """Implicit Quantile Network Q function factory class.

    References:
        * `Dabney et al., Implicit quantile networks for distributional
          reinforcement learning. <https://arxiv.org/abs/1806.06923>`_

    Args:
        n_quantiles (int): the number of quantiles.
        n_greedy_quantiles (int): the number of quantiles for inference.
        embed_size (int): the embedding size.

    """

    TYPE: ClassVar[str] = "iqn"
    _n_quantiles: int
    _n_greedy_quantiles: int
    _embed_size: int

    def __init__(
        self,
        n_quantiles: int = 64,
        n_greedy_quantiles: int = 32,
        embed_size: int = 64,
    ):
        self._n_quantiles = n_quantiles
        self._n_greedy_quantiles = n_greedy_quantiles
        self._embed_size = embed_size

    def create(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        action_size: Optional[int] = None,
    ) -> Union[ContinuousIQNQFunction, DiscreteIQNQFunction]:
        q_func: Union[DiscreteIQNQFunction, ContinuousIQNQFunction]
        if action_size is None:
            assert isinstance(encoder, EncoderWithAction)
            q_func = ContinuousIQNQFunction(
                encoder=encoder,
                n_quantiles=self._n_quantiles,
                n_greedy_quantiles=self._n_greedy_quantiles,
                embed_size=self._embed_size,
            )
        else:
            assert isinstance(encoder, Encoder)
            q_func = DiscreteIQNQFunction(
                encoder=encoder,
                action_size=action_size,
                n_quantiles=self._n_quantiles,
                n_greedy_quantiles=self._n_greedy_quantiles,
                embed_size=self._embed_size,
            )
        return q_func

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "n_quantiles": self._n_quantiles,
            "n_greedy_quantiles": self._n_greedy_quantiles,
            "embed_size": self._embed_size,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    @property
    def n_greedy_quantiles(self) -> int:
        return self._n_greedy_quantiles

    @property
    def embed_size(self) -> int:
        return self._embed_size


class FQFQFunctionFactory(QFunctionFactory):
    """Fully parameterized Quantile Function Q function factory.

    References:
        * `Yang et al., Fully parameterized quantile function for
          distributional reinforcement learning.
          <https://arxiv.org/abs/1911.02140>`_

    Args:
        n_quantiles (int): the number of quantiles.
        embed_size (int): the embedding size.
        entropy_coeff (float): the coefficiency of entropy penalty term.

    """

    TYPE: ClassVar[str] = "fqf"
    _n_quantiles: int
    _embed_size: int
    _entropy_coeff: float

    def __init__(
        self,
        n_quantiles: int = 32,
        embed_size: int = 64,
        entropy_coeff: float = 0.0,
    ):
        self._n_quantiles = n_quantiles
        self._embed_size = embed_size
        self._entropy_coeff = entropy_coeff

    def create(
        self,
        encoder: Union[Encoder, EncoderWithAction],
        action_size: Optional[int] = None,
    ) -> Union[ContinuousFQFQFunction, DiscreteFQFQFunction]:
        q_func: Union[ContinuousFQFQFunction, DiscreteFQFQFunction]
        if action_size is None:
            assert isinstance(encoder, EncoderWithAction)
            q_func = ContinuousFQFQFunction(
                encoder=encoder,
                n_quantiles=self._n_quantiles,
                embed_size=self._embed_size,
                entropy_coeff=self._entropy_coeff,
            )
        else:
            assert isinstance(encoder, Encoder)
            q_func = DiscreteFQFQFunction(
                encoder=encoder,
                action_size=action_size,
                n_quantiles=self._n_quantiles,
                embed_size=self._embed_size,
                entropy_coeff=self._entropy_coeff,
            )
        return q_func

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "n_quantiles": self._n_quantiles,
            "embed_size": self._embed_size,
            "entropy_coeff": self._entropy_coeff,
        }

    @property
    def n_quantiles(self) -> int:
        return self._n_quantiles

    @property
    def embed_size(self) -> int:
        return self._embed_size

    @property
    def entropy_coeff(self) -> float:
        return self._entropy_coeff


Q_FUNC_LIST: Dict[str, Type[QFunctionFactory]] = {}


def register_q_func_factory(cls: Type[QFunctionFactory]) -> None:
    """Registers Q function factory class.

    Args:
        cls (type): Q function factory class inheriting ``QFunctionFactory``.

    """
    is_registered = cls.TYPE in Q_FUNC_LIST
    assert not is_registered, "%s seems to be already registered" % cls.TYPE
    Q_FUNC_LIST[cls.TYPE] = cls


def create_q_func_factory(
    name: str, **kwargs: Dict[str, Any]
) -> QFunctionFactory:
    """Returns registered Q function factory object.

    Args:
        name (str): registered Q function factory type name.
        kwargs (any): Q function arguments.

    Returns:
        d3rlpy.q_functions.QFunctionFactory: Q function factory object.

    """
    assert name in Q_FUNC_LIST, "%s seems not to be registered." % name
    factory = Q_FUNC_LIST[name](**kwargs)  # type: ignore
    assert isinstance(factory, QFunctionFactory)
    return factory


register_q_func_factory(MeanQFunctionFactory)
register_q_func_factory(QRQFunctionFactory)
register_q_func_factory(IQNQFunctionFactory)
register_q_func_factory(FQFQFunctionFactory)
