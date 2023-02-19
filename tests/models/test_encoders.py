import pytest

from d3rlpy.models.encoders import (
    DefaultEncoderFactory,
    DenseEncoderFactory,
    PixelEncoderFactory,
    VectorEncoderFactory,
)
from d3rlpy.models.torch.encoders import (
    PixelEncoder,
    PixelEncoderWithAction,
    VectorEncoder,
    VectorEncoderWithAction,
)


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

    assert factory.get_type() == "pixel"

    # check serization and deserialization
    PixelEncoderFactory.deserialize(factory.serialize())


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

    assert factory.get_type() == "vector"

    # check serization and deserialization
    new_factory = VectorEncoderFactory.deserialize(factory.serialize())
    assert new_factory.hidden_units == factory.hidden_units
    assert new_factory.use_batch_norm == factory.use_batch_norm


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

    assert factory.get_type() == "default"

    # check serization and deserialization
    DefaultEncoderFactory.deserialize(factory.serialize())


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

    assert factory.get_type() == "dense"

    # check serization and deserialization
    DenseEncoderFactory.deserialize(factory.serialize())
