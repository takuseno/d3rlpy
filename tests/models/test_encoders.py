import pytest

from d3rlpy.models.torch.encoders import PixelEncoder
from d3rlpy.models.torch.encoders import PixelEncoderWithAction
from d3rlpy.models.torch.encoders import VectorEncoder
from d3rlpy.models.torch.encoders import VectorEncoderWithAction
from d3rlpy.models.encoders import create_encoder_factory
from d3rlpy.models.encoders import PixelEncoderFactory
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.encoders import DenseEncoderFactory


@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("discrete_action", [False, True])
def test_pixel_encoder_factory(observation_shape, action_size, discrete_action):
    factory = PixelEncoderFactory()

    # test state encoder
    encoder = factory.create(observation_shape)
    assert isinstance(encoder, PixelEncoder)

    # test state-action encoder
    encoder = factory.create_with_action(
        observation_shape, action_size, discrete_action
    )
    assert isinstance(encoder, PixelEncoderWithAction)
    assert encoder._discrete_action == discrete_action

    # test get_params
    assert factory.get_type() == "pixel"
    params = factory.get_params()
    new_factory = PixelEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("discrete_action", [False, True])
def test_vector_encoder_factory(
    observation_shape, action_size, discrete_action
):
    factory = VectorEncoderFactory()

    # test state encoder
    encoder = factory.create(observation_shape)
    assert isinstance(encoder, VectorEncoder)

    # test state-action encoder
    encoder = factory.create_with_action(
        observation_shape, action_size, discrete_action
    )
    assert isinstance(encoder, VectorEncoderWithAction)
    assert encoder._discrete_action == discrete_action

    # test get params
    assert factory.get_type() == "vector"
    params = factory.get_params()
    new_factory = VectorEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("discrete_action", [False, True])
def test_default_encoder_factory(
    observation_shape, action_size, discrete_action
):
    factory = DefaultEncoderFactory()

    # test state encoder
    encoder = factory.create(observation_shape)
    if len(observation_shape) == 3:
        assert isinstance(encoder, PixelEncoder)
    else:
        assert isinstance(encoder, VectorEncoder)

    # test state-action encoder
    encoder = factory.create_with_action(
        observation_shape, action_size, discrete_action
    )
    if len(observation_shape) == 3:
        assert isinstance(encoder, PixelEncoderWithAction)
    else:
        assert isinstance(encoder, VectorEncoderWithAction)
    assert encoder._discrete_action == discrete_action

    # test get_params
    assert factory.get_type() == "default"
    params = factory.get_params()
    new_factory = DefaultEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("discrete_action", [False, True])
def test_dense_encoder_factory(observation_shape, action_size, discrete_action):
    factory = DenseEncoderFactory()

    # test state encoder
    encoder = factory.create(observation_shape)
    assert isinstance(encoder, VectorEncoder)
    assert encoder._use_dense

    # test state-action encoder
    encoder = factory.create_with_action(
        observation_shape, action_size, discrete_action
    )
    assert isinstance(encoder, VectorEncoderWithAction)
    assert encoder._discrete_action == discrete_action
    assert encoder._use_dense

    # test get_params
    assert factory.get_type() == "dense"
    params = factory.get_params()
    new_factory = DenseEncoderFactory(**params)
    assert new_factory.get_params() == params


@pytest.mark.parametrize("name", ["pixel", "vector", "default", "dense"])
def test_create_encoder_factory(name):
    factory = create_encoder_factory(name)
    if name == "pixel":
        assert isinstance(factory, PixelEncoderFactory)
    elif name == "vector":
        assert isinstance(factory, VectorEncoderFactory)
    elif name == "default":
        assert isinstance(factory, DefaultEncoderFactory)
    elif name == "dense":
        assert isinstance(factory, DenseEncoderFactory)
