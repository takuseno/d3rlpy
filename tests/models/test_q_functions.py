from typing import Sequence

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
from d3rlpy.models.torch.encoders import Encoder, EncoderWithAction


def _create_encoder(observation_shape: Sequence[int]) -> Encoder:
    factory = VectorEncoderFactory()
    return factory.create(observation_shape)


def _create_encoder_with_action(
    observation_shape: Sequence[int], action_size: int
) -> EncoderWithAction:
    factory = VectorEncoderFactory()
    return factory.create_with_action(observation_shape, action_size)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
def test_mean_q_function_factory(
    observation_shape: Sequence[int], action_size: int
) -> None:
    factory = MeanQFunctionFactory()
    assert factory.get_type() == "mean"

    encoder_with_action = _create_encoder_with_action(
        observation_shape, action_size
    )
    q_func = factory.create_continuous(encoder_with_action)
    assert isinstance(q_func, ContinuousMeanQFunction)

    encoder = _create_encoder(observation_shape)
    discrete_q_func = factory.create_discrete(encoder, action_size)
    assert isinstance(discrete_q_func, DiscreteMeanQFunction)

    # check serization and deserialization
    MeanQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
def test_qr_q_function_factory(
    observation_shape: Sequence[int], action_size: int
) -> None:
    factory = QRQFunctionFactory()
    assert factory.get_type() == "qr"

    encoder_with_action = _create_encoder_with_action(
        observation_shape, action_size
    )
    q_func = factory.create_continuous(encoder_with_action)
    assert isinstance(q_func, ContinuousQRQFunction)

    encoder = _create_encoder(observation_shape)
    discrete_q_func = factory.create_discrete(encoder, action_size)
    assert isinstance(discrete_q_func, DiscreteQRQFunction)

    # check serization and deserialization
    QRQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
def test_iqn_q_function_factory(
    observation_shape: Sequence[int], action_size: int
) -> None:
    factory = IQNQFunctionFactory()
    assert factory.get_type() == "iqn"

    encoder_with_action = _create_encoder_with_action(
        observation_shape, action_size
    )
    q_func = factory.create_continuous(encoder_with_action)
    assert isinstance(q_func, ContinuousIQNQFunction)

    encoder = _create_encoder(observation_shape)
    discrete_q_func = factory.create_discrete(encoder, action_size)
    assert isinstance(discrete_q_func, DiscreteIQNQFunction)

    # check serization and deserialization
    IQNQFunctionFactory.deserialize(factory.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
def test_fqf_q_function_factory(
    observation_shape: Sequence[int], action_size: int
) -> None:
    factory = FQFQFunctionFactory()
    assert factory.get_type() == "fqf"

    encoder_with_action = _create_encoder_with_action(
        observation_shape, action_size
    )
    q_func = factory.create_continuous(encoder_with_action)
    assert isinstance(q_func, ContinuousFQFQFunction)

    encoder = _create_encoder(observation_shape)
    discrete_q_func = factory.create_discrete(encoder, action_size)
    assert isinstance(discrete_q_func, DiscreteFQFQFunction)

    # check serization and deserialization
    FQFQFunctionFactory.deserialize(factory.serialize())
