import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, ClassVar, Dict, List, Optional, Union, Type
from typing import Sequence

import torch

from .models.torch import Encoder, EncoderWithAction
from .models.torch import PixelEncoder
from .models.torch import PixelEncoderWithAction
from .models.torch import VectorEncoder
from .models.torch import VectorEncoderWithAction


def _create_activation(
    activation_type: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation_type == "relu":
        return torch.relu  # type: ignore
    elif activation_type == "tanh":
        return torch.tanh  # type: ignore
    elif activation_type == "swish":
        return lambda x: x * torch.sigmoid(x)
    raise ValueError("invalid activation_type.")


class EncoderFactory(metaclass=ABCMeta):
    TYPE: ClassVar[str] = "none"

    @abstractmethod
    def create(
        self,
        observation_shape: Sequence[int],
        action_size: Optional[int] = None,
        discrete_action: bool = False,
    ) -> Union[Encoder, EncoderWithAction]:
        """Returns PyTorch's enocder module.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): action size. If None, the encoder does not take
                action as input.
            discrete_action (bool): flag if action-space is discrete.

        Returns:
            torch.nn.Module: an enocder object.

        """

    def get_type(self) -> str:
        """Returns encoder type.

        Returns:
            str: encoder type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """Returns encoder parameters.

        Args:
            deep (bool): flag to deeply copy the parameters.

        Returns:
            dict: encoder parameters.

        """


class PixelEncoderFactory(EncoderFactory):
    """Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE: ClassVar[str] = "pixel"
    _filters: List[Sequence[int]]
    _feature_size: int
    _activation: str
    _use_batch_norm: bool

    def __init__(
        self,
        filters: Optional[List[Sequence[int]]] = None,
        feature_size: int = 512,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        if filters is None:
            self._filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        else:
            self._filters = filters
        self._feature_size = feature_size
        self._activation = activation
        self._use_batch_norm = use_batch_norm

    def create(
        self,
        observation_shape: Sequence[int],
        action_size: Optional[int] = None,
        discrete_action: bool = False,
    ) -> Union[PixelEncoder, PixelEncoderWithAction]:
        assert len(observation_shape) == 3
        activation_fn = _create_activation(self._activation)
        encoder: Union[PixelEncoder, PixelEncoderWithAction]
        if action_size is not None:
            encoder = PixelEncoderWithAction(
                observation_shape=observation_shape,
                action_size=action_size,
                filters=self._filters,
                feature_size=self._feature_size,
                use_batch_norm=self._use_batch_norm,
                discrete_action=discrete_action,
                activation=activation_fn,
            )
        else:
            encoder = PixelEncoder(
                observation_shape=observation_shape,
                filters=self._filters,
                feature_size=self._feature_size,
                use_batch_norm=self._use_batch_norm,
                activation=activation_fn,
            )
        return encoder

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            filters = copy.deepcopy(self._filters)
        else:
            filters = self._filters
        params = {
            "filters": filters,
            "feature_size": self._feature_size,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
        }
        return params


class VectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_dense (bool): flag to use DenseNet architecture.

    """

    TYPE: ClassVar[str] = "vector"
    _hidden_units: Sequence[int]
    _activation: str
    _use_batch_norm: bool
    _use_dense: bool

    def __init__(
        self,
        hidden_units: Optional[Sequence[int]] = None,
        activation: str = "relu",
        use_batch_norm: bool = False,
        use_dense: bool = False,
    ):
        if hidden_units is None:
            self._hidden_units = [256, 256]
        else:
            self._hidden_units = hidden_units
        self._activation = activation
        self._use_batch_norm = use_batch_norm
        self._use_dense = use_dense

    def create(
        self,
        observation_shape: Sequence[int],
        action_size: Optional[int] = None,
        discrete_action: bool = False,
    ) -> Union[VectorEncoder, VectorEncoderWithAction]:
        assert len(observation_shape) == 1
        activation_fn = _create_activation(self._activation)
        encoder: Union[VectorEncoder, VectorEncoderWithAction]
        if action_size is not None:
            encoder = VectorEncoderWithAction(
                observation_shape=observation_shape,
                action_size=action_size,
                hidden_units=self._hidden_units,
                use_batch_norm=self._use_batch_norm,
                use_dense=self._use_dense,
                discrete_action=discrete_action,
                activation=activation_fn,
            )
        else:
            encoder = VectorEncoder(
                observation_shape=observation_shape,
                hidden_units=self._hidden_units,
                use_batch_norm=self._use_batch_norm,
                use_dense=self._use_dense,
                activation=activation_fn,
            )
        return encoder

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        if deep:
            hidden_units = copy.deepcopy(self._hidden_units)
        else:
            hidden_units = self._hidden_units
        params = {
            "hidden_units": hidden_units,
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
            "use_dense": self._use_dense,
        }
        return params


class DefaultEncoderFactory(EncoderFactory):
    """Default encoder factory class.

    This encoder factory returns an encoder based on observation shape.

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE: ClassVar[str] = "default"
    _activation: str
    _use_batch_norm: bool

    def __init__(self, activation: str = "relu", use_batch_norm: bool = False):
        self._activation = activation
        self._use_batch_norm = use_batch_norm

    def create(
        self,
        observation_shape: Sequence[int],
        action_size: Optional[int] = None,
        discrete_action: bool = False,
    ) -> Union[Encoder, EncoderWithAction]:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self._activation, use_batch_norm=self._use_batch_norm
            )
        else:
            factory = VectorEncoderFactory(
                activation=self._activation, use_batch_norm=self._use_batch_norm
            )
        return factory.create(observation_shape, action_size, discrete_action)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
        }


class DenseEncoderFactory(EncoderFactory):
    """DenseNet encoder factory class.

    This is an alias for DenseNet architecture proposed in D2RL.
    This class does exactly same as follows.

    .. code-block:: python

       from d3rlpy.encoders import VectorEncoderFactory

       factory = VectorEncoderFactory(hidden_units=[256, 256, 256, 256],
                                      use_dense=True)

    For now, this only supports vector observations.

    References:
        * `Sinha et al., D2RL: Deep Dense Architectures in Reinforcement
          Learning. <https://arxiv.org/abs/2010.09163>`_

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE: ClassVar[str] = "dense"
    _activation: str
    _use_batch_norm: bool

    def __init__(self, activation: str = "relu", use_batch_norm: bool = False):
        self._activation = activation
        self._use_batch_norm = use_batch_norm

    def create(
        self,
        observation_shape: Sequence[int],
        action_size: Optional[int] = None,
        discrete_action: bool = False,
    ) -> Union[VectorEncoder, VectorEncoderWithAction]:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self._activation,
            use_dense=True,
            use_batch_norm=self._use_batch_norm,
        )
        return factory.create(observation_shape, action_size, discrete_action)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        return {
            "activation": self._activation,
            "use_batch_norm": self._use_batch_norm,
        }


ENCODER_LIST: Dict[str, Type[EncoderFactory]] = {}


def register_encoder_factory(cls: Type[EncoderFactory]) -> None:
    """Registers encoder factory class.

    Args:
        cls (type): encoder factory class inheriting ``EncoderFactory``.

    """
    is_registered = cls.TYPE in ENCODER_LIST
    assert not is_registered, "%s seems to be already registered" % cls.TYPE
    ENCODER_LIST[cls.TYPE] = cls


def create_encoder_factory(
    name: str, **kwargs: Dict[str, Any]
) -> EncoderFactory:
    """Returns registered encoder factory object.

    Args:
        name (str): regsitered encoder factory type name.
        kwargs (any): encoder arguments.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    assert name in ENCODER_LIST, "%s seems not to be registered." % name
    factory = ENCODER_LIST[name](**kwargs)  # type: ignore
    assert isinstance(factory, EncoderFactory)
    return factory


register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
register_encoder_factory(DenseEncoderFactory)
