from typing import Sequence

import gym
import numpy as np
import pytest
import torch

from d3rlpy.dataset import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    EpisodeGenerator,
)
from d3rlpy.preprocessing import MinMaxActionScaler


@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_action_scaler(action_size: int, batch_size: int) -> None:
    actions = np.random.random((batch_size, action_size)).astype("f4")

    maximum = actions.max(axis=0)
    minimum = actions.min(axis=0)

    scaler = MinMaxActionScaler(maximum=maximum, minimum=minimum)
    assert scaler.built
    assert scaler.get_type() == "min_max"

    # check range
    y = scaler.transform(torch.tensor(actions))
    assert np.all(y.numpy() >= -1.0)
    assert np.all(y.numpy() <= 1.0)

    # check transorm
    x = torch.rand((batch_size, action_size))
    y = scaler.transform(x)
    ref_y = (x.numpy() - minimum.reshape((1, -1))) / (
        maximum - minimum
    ).reshape((1, -1))
    assert np.allclose(y.numpy(), ref_y * 2.0 - 1.0)

    # check reverse_transorm
    assert torch.allclose(scaler.reverse_transform(y), x, atol=1e-6)

    # check transform_numpy
    y = scaler.transform_numpy(x.numpy())
    assert np.allclose(y, ref_y * 2.0 - 1.0, atol=1e-6)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), x.numpy(), atol=1e-6)

    # check serialization and deserialization
    new_scaler = MinMaxActionScaler.deserialize(scaler.serialize())
    assert np.all(new_scaler.minimum == scaler.minimum)
    assert np.all(new_scaler.maximum == scaler.maximum)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_action_scaler_with_transition_picker(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size)).astype("f4")
    rewards = np.random.random(batch_size)
    terminals = np.zeros(batch_size)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = actions.max(axis=0)
    minimum = actions.min(axis=0)

    scaler = MinMaxActionScaler()
    assert not scaler.built
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert np.allclose(scaler.minimum, minimum)
    assert np.allclose(scaler.maximum, maximum)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_action_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size)).astype("f4")
    rewards = np.random.random(batch_size)
    terminals = np.zeros(batch_size)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = actions.max(axis=0)
    minimum = actions.min(axis=0)

    scaler = MinMaxActionScaler()
    assert not scaler.built
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert np.allclose(scaler.minimum, minimum)
    assert np.allclose(scaler.maximum, maximum)


def test_min_max_action_scaler_with_env() -> None:
    env = gym.make("Pendulum-v1")

    scaler = MinMaxActionScaler()
    assert not scaler.built
    scaler.fit_with_env(env)
    assert scaler.built

    assert np.all(scaler.minimum == env.action_space.low)
    assert np.all(scaler.maximum == env.action_space.high)
