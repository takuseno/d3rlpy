import pytest

from d3rlpy.models.torch.encoders import PixelEncoder
from d3rlpy.models.torch.encoders import PixelEncoderWithAction
from d3rlpy.models.torch.encoders import VectorEncoder
from d3rlpy.models.torch.encoders import VectorEncoderWithAction
from d3rlpy.encoders import create_encoder_factory
from d3rlpy.encoders import PixelEncoderFactory
from d3rlpy.encoders import VectorEncoderFactory
from d3rlpy.encoders import DefaultEncoderFactory
from d3rlpy.encoders import DenseEncoderFactory


@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
@pytest.mark.parametrize('action_size', [None, 2])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_pixel_encoder_factory(observation_shape, action_size,
                               discrete_action):
    factory = PixelEncoderFactory()

    encoder = factory.create(observation_shape, action_size, discrete_action)

    if action_size is None:
        assert isinstance(encoder, PixelEncoder)
    else:
        assert isinstance(encoder, PixelEncoderWithAction)
        assert encoder._discrete_action == discrete_action

    assert factory.get_type() == 'pixel'

    params = factory.get_params()
    new_factory = PixelEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_vector_encoder_factory(observation_shape, action_size,
                                discrete_action):
    factory = VectorEncoderFactory()

    encoder = factory.create(observation_shape, action_size, discrete_action)

    if action_size is None:
        assert isinstance(encoder, VectorEncoder)
    else:
        assert isinstance(encoder, VectorEncoderWithAction)
        assert encoder._discrete_action == discrete_action

    assert factory.get_type() == 'vector'

    params = factory.get_params()
    new_factory = VectorEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [None, 2])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_default_encoder_factory(observation_shape, action_size,
                                 discrete_action):
    factory = DefaultEncoderFactory()

    encoder = factory.create(observation_shape, action_size, discrete_action)

    if len(observation_shape) == 3:
        if action_size is None:
            assert isinstance(encoder, PixelEncoder)
        else:
            assert isinstance(encoder, PixelEncoderWithAction)
    else:
        if action_size is None:
            assert isinstance(encoder, VectorEncoder)
        else:
            assert isinstance(encoder, VectorEncoderWithAction)
    if action_size is not None:
        assert encoder._discrete_action == discrete_action

    assert factory.get_type() == 'default'

    params = factory.get_params()
    new_factory = DefaultEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [None, 2])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_dense_encoder_factory(observation_shape, action_size,
                               discrete_action):
    factory = DenseEncoderFactory()

    encoder = factory.create(observation_shape, action_size, discrete_action)

    if action_size is None:
        assert isinstance(encoder, VectorEncoder)
    else:
        assert isinstance(encoder, VectorEncoderWithAction)
        assert encoder._discrete_action == discrete_action
    assert encoder._use_dense

    assert factory.get_type() == 'dense'

    params = factory.get_params()
    new_factory = DenseEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize('name', ['pixel', 'vector', 'default', 'dense'])
def test_create_encoder_factory(name):
    factory = create_encoder_factory(name)
    if name == 'pixel':
        assert isinstance(factory, PixelEncoderFactory)
    elif name == 'vector':
        assert isinstance(factory, VectorEncoderFactory)
    elif name == 'default':
        assert isinstance(factory, DefaultEncoderFactory)
    elif name == 'dense':
        assert isinstance(factory, DenseEncoderFactory)
