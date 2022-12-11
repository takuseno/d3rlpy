from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from dataclasses_json import config
from torch import nn

from ..dataset import Shape, cast_flat_shape
from ..serializable_config import SerializableConfig
from ..torch_utility import Swish
from .torch import (
    Encoder,
    EncoderWithAction,
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)

__all__ = [
    "EncoderFactory",
    "PixelEncoderFactory",
    "VectorEncoderFactory",
    "DefaultEncoderFactory",
    "DenseEncoderFactory",
    "ENCODER_LIST",
    "register_encoder_factory",
    "create_encoder_factory",
    "make_encoder_field",
]


def _create_activation(activation_type: str) -> nn.Module:
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "swish":
        return Swish()
    raise ValueError("invalid activation_type.")


class EncoderFactory(SerializableConfig):
    def create(self, observation_shape: Shape) -> Encoder:
        """Returns PyTorch's state enocder module.

        Args:
            observation_shape: observation shape.

        Returns:
            an enocder object.

        """
        raise NotImplementedError

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        """Returns PyTorch's state-action enocder module.

        Args:
            observation_shape: observation shape.
            action_size: action size. If None, the encoder does not take
                action as input.
            discrete_action: flag if action-space is discrete.

        Returns:
            an enocder object.

        """
        raise NotImplementedError

    @staticmethod
    def get_type() -> str:
        """Returns encoder type.

        Returns:
            encoder type.

        """
        raise NotImplementedError


@dataclass(frozen=True)
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
        dropout_rate (float): dropout probability.

    """

    filters: List[Sequence[int]] = field(
        default_factory=lambda: [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    )
    feature_size: int = 512
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None

    def create(self, observation_shape: Shape) -> PixelEncoder:
        assert len(observation_shape) == 3
        return PixelEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            filters=self.filters,
            feature_size=self.feature_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            activation=_create_activation(self.activation),
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> PixelEncoderWithAction:
        assert len(observation_shape) == 3
        return PixelEncoderWithAction(
            observation_shape=cast_flat_shape(observation_shape),
            action_size=action_size,
            filters=self.filters,
            feature_size=self.feature_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            discrete_action=discrete_action,
            activation=_create_activation(self.activation),
        )

    @staticmethod
    def get_type() -> str:
        return "pixel"


@dataclass(frozen=True)
class VectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        use_dense (bool): flag to use DenseNet architecture.
        dropout_rate (float): dropout probability.

    """

    hidden_units: Sequence[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None
    use_dense: bool = False

    def create(self, observation_shape: Shape) -> VectorEncoder:
        assert len(observation_shape) == 1
        return VectorEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            hidden_units=self.hidden_units,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            use_dense=self.use_dense,
            activation=_create_activation(self.activation),
        )

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        assert len(observation_shape) == 1
        return VectorEncoderWithAction(
            observation_shape=cast_flat_shape(observation_shape),
            action_size=action_size,
            hidden_units=self.hidden_units,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            use_dense=self.use_dense,
            discrete_action=discrete_action,
            activation=_create_activation(self.activation),
        )

    @staticmethod
    def get_type() -> str:
        return "vector"


@dataclass(frozen=True)
class DefaultEncoderFactory(EncoderFactory):
    """Default encoder factory class.

    This encoder factory returns an encoder based on observation shape.

    Args:
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.
        dropout_rate (float): dropout probability.

    """

    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None

    def create(self, observation_shape: Shape) -> Encoder:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> EncoderWithAction:
        factory: Union[PixelEncoderFactory, VectorEncoderFactory]
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        else:
            factory = VectorEncoderFactory(
                activation=self.activation,
                use_batch_norm=self.use_batch_norm,
                dropout_rate=self.dropout_rate,
            )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    @staticmethod
    def get_type() -> str:
        return "default"


@dataclass(frozen=True)
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
        dropout_rate (float): dropout probability.

    """

    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None

    def create(self, observation_shape: Shape) -> VectorEncoder:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self.activation,
            use_dense=True,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )
        return factory.create(observation_shape)

    def create_with_action(
        self,
        observation_shape: Shape,
        action_size: int,
        discrete_action: bool = False,
    ) -> VectorEncoderWithAction:
        if len(observation_shape) == 3:
            raise NotImplementedError("pixel observation is not supported.")
        factory = VectorEncoderFactory(
            hidden_units=[256, 256, 256, 256],
            activation=self.activation,
            use_dense=True,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
        )
        return factory.create_with_action(
            observation_shape, action_size, discrete_action
        )

    @staticmethod
    def get_type() -> str:
        return "dense"


ENCODER_LIST: Dict[str, Type[EncoderFactory]] = {}


def register_encoder_factory(cls: Type[EncoderFactory]) -> None:
    """Registers encoder factory class.

    Args:
        cls: encoder factory class inheriting ``EncoderFactory``.

    """
    type_name = cls.get_type()
    is_registered = type_name in ENCODER_LIST
    assert not is_registered, f"{type_name} seems to be already registered"
    ENCODER_LIST[type_name] = cls


def create_encoder_factory(name: str, **kwargs: Any) -> EncoderFactory:
    """Returns registered encoder factory object.

    Args:
        name: regsitered encoder factory type name.
        kwargs: encoder arguments.

    Returns:
        encoder factory object.

    """
    assert name in ENCODER_LIST, f"{name} seems not to be registered."
    factory = ENCODER_LIST[name](**kwargs)
    assert isinstance(factory, EncoderFactory)
    return factory


def _encoder(encoder: EncoderFactory) -> Dict[str, Any]:
    return {"type": encoder.get_type(), "params": asdict(encoder)}


def _decoder(dict_config: Dict[str, Any]) -> EncoderFactory:
    return create_encoder_factory(dict_config["type"], **dict_config["params"])


def make_encoder_field() -> EncoderFactory:
    return field(
        metadata=config(encoder=_encoder, decoder=_decoder),
        default=DefaultEncoderFactory(),
    )


register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
register_encoder_factory(DenseEncoderFactory)
