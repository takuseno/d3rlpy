import pytest

from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import (
    FQFQFunctionFactory,
    IQNQFunctionFactory,
    MeanQFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.models.torch import (
    ContinuousFQFQFunction,
    ContinuousIQNQFunction,
    ContinuousMeanQFunction,
    ContinuousQRQFunction,
    DiscreteFQFQFunction,
    DiscreteIQNQFunction,
    DiscreteMeanQFunction,
    DiscreteQRQFunction,
)


def _create_encoder(observation_shape, action_size):
    factory = VectorEncoderFactory()
    if action_size is None:
        encoder = factory.create_with_action(observation_shape, 2)
    else:
        encoder = factory.create(observation_shape)
    return encoder


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [None, 2])
def test_mean_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = MeanQFunctionFactory()
    if action_size is None:
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousMeanQFunction)
    else:
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteMeanQFunction)

    assert factory.get_type() == "mean"

    # check serization and deserialization
    MeanQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [None, 2])
def test_qr_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = QRQFunctionFactory()
    if action_size is None:
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousQRQFunction)
    else:
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteQRQFunction)

    assert factory.get_type() == "qr"

    # check serization and deserialization
    QRQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [None, 2])
def test_iqn_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = IQNQFunctionFactory()
    if action_size is None:
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousIQNQFunction)
    else:
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteIQNQFunction)

    assert factory.get_type() == "iqn"

    # check serization and deserialization
    IQNQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [None, 2])
def test_fqf_q_function_factory(observation_shape, action_size):
    encoder = _create_encoder(observation_shape, action_size)

    factory = FQFQFunctionFactory()
    if action_size is None:
        q_func = factory.create_continuous(encoder)
        assert isinstance(q_func, ContinuousFQFQFunction)
    else:
        q_func = factory.create_discrete(encoder, action_size)
        assert isinstance(q_func, DiscreteFQFQFunction)

    assert factory.get_type() == "fqf"

    # check serization and deserialization
    FQFQFunctionFactory.deserialize(factory.serialize())
