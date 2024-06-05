from dataclasses import dataclass, field
from typing import List, Optional, Union

from ..dataset import cast_flat_shape
from ..serializable_config import DynamicConfig, generate_config_registration
from ..types import Shape
from .torch import (
    Encoder,
    EncoderWithAction,
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)
from .utility import create_activation

__all__ = [
    "EncoderFactory",
    "PixelEncoderFactory",
    "VectorEncoderFactory",
    "DefaultEncoderFactory",
    "register_encoder_factory",
    "make_encoder_field",
]


class EncoderFactory(DynamicConfig):
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


@dataclass()
class PixelEncoderFactory(EncoderFactory):
    """Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): List of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): Last linear layer size.
        activation (str): Activation function name.
        use_batch_norm (bool): Flag to insert batch normalization layers.
        dropout_rate (float): Dropout probability.
        exclude_last_activation (bool): Flag to exclude activation function at
            the last layer.
        last_activation (str): Activation function name for the last layer.
    """

    filters: List[List[int]] = field(
        default_factory=lambda: [[32, 8, 4], [64, 4, 2], [64, 3, 1]]
    )
    feature_size: int = 512
    activation: str = "relu"
    use_batch_norm: bool = False
    dropout_rate: Optional[float] = None
    exclude_last_activation: bool = False
    last_activation: Optional[str] = None

    def create(self, observation_shape: Shape) -> PixelEncoder:
        assert len(observation_shape) == 3
        return PixelEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            filters=self.filters,
            feature_size=self.feature_size,
            use_batch_norm=self.use_batch_norm,
            dropout_rate=self.dropout_rate,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
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
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    @staticmethod
    def get_type() -> str:
        return "pixel"


@dataclass()
class VectorEncoderFactory(EncoderFactory):
    """Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): List of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): Flag to insert batch normalization layers.
        use_layer_norm (bool): Flag to insert layer normalization layers.
        dropout_rate (float): Dropout probability.
        exclude_last_activation (bool): Flag to exclude activation function at
            the last layer.
        last_activation (str): Activation function name for the last layer.
    """

    hidden_units: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    exclude_last_activation: bool = False
    last_activation: Optional[str] = None

    def create(self, observation_shape: Shape) -> VectorEncoder:
        assert len(observation_shape) == 1
        return VectorEncoder(
            observation_shape=cast_flat_shape(observation_shape),
            hidden_units=self.hidden_units,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
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
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            discrete_action=discrete_action,
            activation=create_activation(self.activation),
            exclude_last_activation=self.exclude_last_activation,
            last_activation=(
                create_activation(self.last_activation)
                if self.last_activation
                else None
            ),
        )

    @staticmethod
    def get_type() -> str:
        return "vector"


@dataclass()
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


register_encoder_factory, make_encoder_field = generate_config_registration(
    EncoderFactory, lambda: DefaultEncoderFactory()
)


register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
