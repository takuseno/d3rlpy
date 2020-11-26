import copy
import torch

from abc import ABCMeta, abstractmethod
from d3rlpy.models.torch.encoders import PixelEncoder
from d3rlpy.models.torch.encoders import PixelEncoderWithAction
from d3rlpy.models.torch.encoders import VectorEncoder
from d3rlpy.models.torch.encoders import VectorEncoderWithAction


def _create_activation(activation_type):
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'tanh':
        return torch.tanh
    elif activation_type == 'elu':
        return torch.elu
    elif activation_type == 'swish':
        return lambda x: x * torch.sigmoid(x)
    raise ValueError('invalid activation_type.')


class EncoderFactory(metaclass=ABCMeta):
    TYPE = 'none'

    @abstractmethod
    def create(self,
               observation_shape,
               action_size=None,
               discrete_action=False):
        """ Returns PyTorch's enocder module.

        Args:
            observation_shape (tuple): observation shape.
            action_size (int): action size. If None, the encoder does not take
                action as input.
            discrete_action (bool): flag if action-space is discrete.

        Returns:
            torch.nn.Module: an enocder object.

        """
        pass

    def get_type(self):
        """ Returns encoder type.

        Returns:
            str: encoder type.

        """
        return self.TYPE

    @abstractmethod
    def get_params(self, deep=False):
        """ Returns encoder parameters.

        Args:
            deep (bool): flag to deeply copy the parameters.

        Returns:
            dict: encoder parameters.

        """
        pass


class PixelEncoderFactory(EncoderFactory):
    """ Pixel encoder factory class.

    This is the default encoder factory for image observation.

    Args:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``. If None,
            ``Nature DQN``-based architecture is used.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    Attributes:
        filters (list): list of tuples consisting with
            ``(filter_size, kernel_size, stride)``.
        feature_size (int): the last linear layer size.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE = 'pixel'

    def __init__(self,
                 filters=None,
                 feature_size=512,
                 activation='relu',
                 use_batch_norm=False):
        if filters is None:
            self.filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        else:
            self.filters = filters
        self.feature_size = feature_size
        self.activation = activation
        self.use_batch_norm = use_batch_norm

    def create(self,
               observation_shape,
               action_size=None,
               discrete_action=False):
        assert len(observation_shape) == 3
        activation_fn = _create_activation(self.activation)
        if action_size is not None:
            encoder = PixelEncoderWithAction(
                observation_shape=observation_shape,
                action_size=action_size,
                filters=self.filters,
                feature_size=self.feature_size,
                use_batch_norm=self.use_batch_norm,
                discrete_action=discrete_action,
                activation=activation_fn)
        else:
            encoder = PixelEncoder(observation_shape=observation_shape,
                                   filters=self.filters,
                                   feature_size=self.feature_size,
                                   use_batch_norm=self.use_batch_norm,
                                   activation=activation_fn)
        return encoder

    def get_params(self, deep=False):
        if deep:
            filters = copy.deepcopy(self.filters)
        else:
            filters = self.filters
        params = {
            'filters': filters,
            'feature_size': self.feature_size,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm
        }
        return params


class VectorEncoderFactory(EncoderFactory):
    """ Vector encoder factory class.

    This is the default encoder factory for vector observation.

    Args:
        hidden_units (list): list of hidden unit sizes. If ``None``, the
            standard architecture with ``[256, 256]`` is used.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    Attributes:
        hidden_units (list): list of hidden unit sizes.
        activation (str): activation function name.
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE = 'vector'

    def __init__(self,
                 hidden_units=None,
                 activation='relu',
                 use_batch_norm=False):
        if hidden_units is None:
            self.hidden_units = [256, 256]
        else:
            self.hidden_units = hidden_units
        self.activation = activation
        self.use_batch_norm = use_batch_norm

    def create(self,
               observation_shape,
               action_size=None,
               discrete_action=False):
        assert len(observation_shape) == 1
        activation_fn = _create_activation(self.activation)
        if action_size is not None:
            encoder = VectorEncoderWithAction(
                observation_shape=observation_shape,
                action_size=action_size,
                hidden_units=self.hidden_units,
                use_batch_norm=self.use_batch_norm,
                discrete_action=discrete_action,
                activation=activation_fn)
        else:
            encoder = VectorEncoder(observation_shape=observation_shape,
                                    hidden_units=self.hidden_units,
                                    use_batch_norm=self.use_batch_norm,
                                    activation=activation_fn)
        return encoder

    def get_params(self, deep=False):
        if deep:
            hidden_units = copy.deepcopy(self.hidden_units)
        else:
            hidden_units = self.hidden_units
        params = {
            'hidden_units': hidden_units,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm
        }
        return params


class DefaultEncoderFactory(EncoderFactory):
    """ Default encoder factory class.

    This encoder factory returns an encoder based on observation shape.

    Args:
        use_batch_norm (bool): flag to insert batch normalization layers.

    Attributes:
        use_batch_norm (bool): flag to insert batch normalization layers.

    """

    TYPE = 'default'

    def __init__(self, use_batch_norm=False):
        self.use_batch_norm = use_batch_norm

    def create(self,
               observation_shape,
               action_size=None,
               discrete_action=False):
        if len(observation_shape) == 3:
            factory = PixelEncoderFactory(use_batch_norm=self.use_batch_norm)
        else:
            factory = VectorEncoderFactory(use_batch_norm=self.use_batch_norm)
        return factory.create(observation_shape, action_size, discrete_action)

    def get_params(self, deep=False):
        return {'use_batch_norm': self.use_batch_norm}


ENCODER_LIST = {}


def register_encoder_factory(cls):
    """ Registers encoder factory class.

    Args:
        cls (type): encoder factory class inheriting ``EncoderFactory``.

    """
    is_registered = cls.TYPE in ENCODER_LIST
    assert not is_registered, '%s seems to be already registered' % cls.TYPE
    ENCODER_LIST[cls.TYPE] = cls


def create_encoder_factory(name, **kwargs):
    """ Returns registered encoder factory object.

    Args:
        name (str): regsitered encoder factory type name.
        kwargs (any): encoder arguments.

    Returns:
        d3rlpy.encoders.EncoderFactory: encoder factory object.

    """
    assert name in ENCODER_LIST, '%s seems not to be registered.' % name
    factory = ENCODER_LIST[name](**kwargs)
    assert isinstance(factory, EncoderFactory)
    return factory


register_encoder_factory(VectorEncoderFactory)
register_encoder_factory(PixelEncoderFactory)
register_encoder_factory(DefaultEncoderFactory)
