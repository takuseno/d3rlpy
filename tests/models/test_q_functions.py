from typing import Sequence

import pytest

from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.models.q_functions import (
    IQNQFunctionFactory,
    MeanQFunctionFactory,
    QRQFunctionFactory,
)
from d3rlpy.models.torch import (
    ContinuousIQNQFunction,
    ContinuousIQNQFunctionForwarder,
    ContinuousMeanQFunction,
    ContinuousMeanQFunctionForwarder,
    ContinuousQRQFunction,
    ContinuousQRQFunctionForwarder,
    DiscreteIQNQFunction,
    DiscreteIQNQFunctionForwarder,
    DiscreteMeanQFunction,
    DiscreteMeanQFunctionForwarder,
    DiscreteQFunctionForwarder,
    DiscreteQRQFunction,
    compute_output_size,
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
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder_with_action, "cpu:0"
    )
    q_func, forwarder = factory.create_continuous(
        encoder_with_action, hidden_size
    )
    assert isinstance(q_func, ContinuousMeanQFunction)
    assert isinstance(forwarder, ContinuousMeanQFunctionForwarder)

    encoder = _create_encoder(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder, "cpu:0")
    discrete_q_func, discrete_forwarder = factory.create_discrete(
        encoder, hidden_size, action_size
    )
    assert isinstance(discrete_q_func, DiscreteMeanQFunction)
    assert isinstance(discrete_forwarder, DiscreteMeanQFunctionForwarder)

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
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder_with_action, "cpu:0"
    )
    q_func, forwarder = factory.create_continuous(
        encoder_with_action, hidden_size
    )
    assert isinstance(q_func, ContinuousQRQFunction)
    assert isinstance(forwarder, ContinuousQRQFunctionForwarder)

    encoder = _create_encoder(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder, "cpu:0")
    discrete_q_func, discrete_forwarder = factory.create_discrete(
        encoder, hidden_size, action_size
    )
    assert isinstance(discrete_q_func, DiscreteQRQFunction)
    assert isinstance(discrete_forwarder, DiscreteQFunctionForwarder)

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
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder_with_action, "cpu:0"
    )
    q_func, forwarder = factory.create_continuous(
        encoder_with_action, hidden_size
    )
    assert isinstance(q_func, ContinuousIQNQFunction)
    assert isinstance(forwarder, ContinuousIQNQFunctionForwarder)

    encoder = _create_encoder(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder, "cpu:0")
    discrete_q_func, discrete_forwarder = factory.create_discrete(
        encoder, hidden_size, action_size
    )
    assert isinstance(discrete_q_func, DiscreteIQNQFunction)
    assert isinstance(discrete_forwarder, DiscreteIQNQFunctionForwarder)

    # check serization and deserialization
    IQNQFunctionFactory.deserialize(factory.serialize())
