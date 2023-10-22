from typing import Sequence

import numpy as np
import pytest
import torch

from d3rlpy.dataset import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    EpisodeGenerator,
)
from d3rlpy.preprocessing import (
    MinMaxObservationScaler,
    PixelObservationScaler,
    StandardObservationScaler,
)
from d3rlpy.types import FloatNDArray

from ..dummy_env import DummyAtari


@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
def test_pixel_observation_scaler(observation_shape: Sequence[int]) -> None:
    scaler = PixelObservationScaler()

    x = torch.randint(high=255, size=observation_shape)

    y = scaler.transform(x)

    assert torch.all(y == x.float() / 255.0)

    assert scaler.get_type() == "pixel"
    assert torch.all(scaler.reverse_transform(y) == x)
    assert scaler.built

    # check serialization and deserialization
    PixelObservationScaler.deserialize(scaler.serialize())


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_observation_scaler(
    observation_shape: Sequence[int], batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")

    maximum = observations.max(axis=0)
    minimum = observations.min(axis=0)
    scaler = MinMaxObservationScaler(maximum=maximum, minimum=minimum)
    assert scaler.built
    assert scaler.get_type() == "min_max"

    # check range
    y = scaler.transform(torch.tensor(observations))
    assert np.all(y.numpy() >= -1.0)
    assert np.all(y.numpy() <= 1.0)

    # check transform
    x = torch.rand((batch_size, *observation_shape))
    y = scaler.transform(x)
    ref_y = (x.numpy() - minimum.reshape((1, -1))) / (
        maximum - minimum
    ).reshape((1, -1))
    assert np.allclose(y.numpy(), ref_y * 2.0 - 1.0, atol=1e-6)

    # check reverse_transform
    assert torch.allclose(scaler.reverse_transform(y), x, atol=1e-6)

    # check transform_numpy
    y = scaler.transform_numpy(x.numpy())
    assert np.allclose(y, ref_y * 2.0 - 1.0, atol=1e-6)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), x.numpy(), atol=1e-6)

    # check serialization and deserialization
    new_scaler = MinMaxObservationScaler.deserialize(scaler.serialize())
    assert np.all(new_scaler.minimum == scaler.minimum)
    assert np.all(new_scaler.maximum == scaler.maximum)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_observation_scaler_with_transition_picker(
    observation_shape: Sequence[int], batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1))
    rewards: FloatNDArray = np.random.random(batch_size).astype(np.float32)
    terminals: FloatNDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = observations.max(axis=0)
    minimum = observations.min(axis=0)

    scaler = MinMaxObservationScaler()
    assert not scaler.built
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert np.allclose(scaler.minimum, minimum)
    assert np.allclose(scaler.maximum, maximum)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_observation_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int], batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1))
    rewards: FloatNDArray = np.random.random(batch_size).astype(np.float32)
    terminals: FloatNDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = observations.max(axis=0)
    minimum = observations.min(axis=0)

    scaler = MinMaxObservationScaler()
    assert not scaler.built
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert np.allclose(scaler.minimum, minimum)
    assert np.allclose(scaler.maximum, maximum)


def test_min_max_observation_scaler_with_env() -> None:
    env = DummyAtari()

    scaler = MinMaxObservationScaler()
    assert not scaler.built
    scaler.fit_with_env(env)
    assert scaler.built

    x = torch.tensor(env.reset()[0].reshape((1,) + env.observation_space.shape))
    y = scaler.transform(x)

    assert torch.all(x / 255.0 * 2.0 - 1.0 == y)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_observation_scaler(
    observation_shape: Sequence[int], batch_size: int, eps: float
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardObservationScaler(mean=mean, std=std, eps=eps)
    assert scaler.built
    assert scaler.get_type() == "standard"

    # check transform
    x = torch.rand((batch_size, *observation_shape))
    y = scaler.transform(x)
    ref_y = (x.numpy() - mean.reshape((1, -1))) / (std.reshape((1, -1)) + eps)
    assert np.allclose(y.numpy(), ref_y)

    # check reverse_transform
    assert torch.allclose(scaler.reverse_transform(y), x, atol=1e-6)

    # check transform_numpy
    y = scaler.transform_numpy(x.numpy())
    assert np.allclose(y, ref_y)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), x.numpy(), atol=1e-6)

    # check serialization and deserialization
    new_scaler = StandardObservationScaler.deserialize(scaler.serialize())
    assert np.all(new_scaler.mean == scaler.mean)
    assert np.all(new_scaler.std == scaler.std)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [32])
def test_standard_observation_scaler_with_transition_picker(
    observation_shape: Sequence[int], batch_size: int, eps: float
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1)).astype("f4")
    rewards: FloatNDArray = np.random.random(batch_size).astype(np.float32)
    terminals: FloatNDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardObservationScaler(eps=eps)
    assert not scaler.built
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.mean is not None and scaler.std is not None
    assert np.allclose(scaler.mean, mean)
    assert np.allclose(scaler.std, std)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [32])
def test_standard_observation_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int], batch_size: int, eps: float
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1)).astype("f4")
    rewards: FloatNDArray = np.random.random(batch_size).astype(np.float32)
    terminals: FloatNDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardObservationScaler(eps=eps)
    assert not scaler.built
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.mean is not None and scaler.std is not None
    assert np.allclose(scaler.mean, mean)
    assert np.allclose(scaler.std, std)
