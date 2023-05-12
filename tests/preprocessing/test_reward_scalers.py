from typing import Sequence

import numpy as np
import pytest
import torch

from d3rlpy.dataset import EpisodeGenerator
from d3rlpy.preprocessing import (
    ClipRewardScaler,
    ConstantShiftRewardScaler,
    MinMaxRewardScaler,
    MultiplyRewardScaler,
    ReturnBasedRewardScaler,
    StandardRewardScaler,
)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_multiply_reward_scaler(batch_size: int, multiplier: float) -> None:
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = MultiplyRewardScaler(multiplier)
    assert scaler.built

    # check trnsform
    y = scaler.transform(torch.tensor(rewards))
    assert np.allclose(y.numpy(), rewards * multiplier)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards)

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, rewards * multiplier)

    assert scaler.get_type() == "multiply"

    # check serialization and deserialization
    new_scaler = MultiplyRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.multiplier == scaler.multiplier


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("low", [-0.1])
@pytest.mark.parametrize("high", [0.1])
@pytest.mark.parametrize("multiplier", [10.0])
def test_clip_reward_scaler(
    batch_size: int, low: float, high: float, multiplier: float
) -> None:
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = ClipRewardScaler(low, high, multiplier)
    assert scaler.built

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() <= multiplier * 0.1)
    assert np.all(y.numpy() >= multiplier * -0.1)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), np.clip(rewards, low, high))

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.all(y == multiplier * np.clip(rewards, low, high))

    assert scaler.get_type() == "clip"

    # check serialization and deserialization
    new_scaler = ClipRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.low == scaler.low
    assert new_scaler.high == scaler.high
    assert new_scaler.multiplier == scaler.multiplier


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_min_max_reward_scaler(batch_size: int, multiplier: float) -> None:
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    maximum = float(rewards.max())
    minimum = float(rewards.min())

    scaler = MinMaxRewardScaler(
        minimum=minimum, maximum=maximum, multiplier=multiplier
    )
    assert scaler.built

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() >= 0.0)
    assert np.all(y.numpy() <= multiplier)

    # check reference value
    ref_y = multiplier * (rewards - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y)

    # check reverse_transform
    ref_x = ref_y * (maximum - minimum) / multiplier + minimum
    assert np.allclose(scaler.reverse_transform(y).numpy(), ref_x)

    # check reverse_transform_numpy
    assert np.allclose(scaler.transform_numpy(rewards), ref_y)

    assert scaler.get_type() == "min_max"

    # check serialization and deserialization
    new_scaler = MinMaxRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.minimum == scaler.minimum
    assert new_scaler.maximum == scaler.maximum


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_reward_scaler_with_episode(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards = np.random.random(batch_size)
    terminals = np.zeros(batch_size)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    rewards_without_first = []
    for episode in episodes:
        rewards_without_first += episode.rewards[1:].tolist()

    maximum = np.max(rewards_without_first)
    minimum = np.min(rewards_without_first)

    scaler = MinMaxRewardScaler()
    assert not scaler.built
    scaler.fit(episodes)
    assert scaler.built

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y, atol=1e-4)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
@pytest.mark.parametrize("multiplier", [10.0])
def test_standard_reward_scaler(
    batch_size: int, eps: float, multiplier: float
) -> None:
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    mean = float(np.mean(rewards))
    std = float(np.std(rewards))

    scaler = StandardRewardScaler(
        mean=mean, std=std, eps=eps, multiplier=multiplier
    )
    assert scaler.built

    # check values
    y = scaler.transform(torch.tensor(rewards))
    ref_y = multiplier * (rewards - mean) / (std + eps)
    assert np.allclose(y.numpy(), ref_y, atol=1e-3)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards, atol=1e-3)

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, ref_y, atol=1e-4)

    assert scaler.get_type() == "standard"

    # check serialization and deserialization
    new_scaler = StandardRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.mean == scaler.mean
    assert new_scaler.std == scaler.std
    assert new_scaler.multiplier == scaler.multiplier


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_reward_scaler_with_episode(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    eps: float,
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards = np.random.random(batch_size).astype("f4")
    terminals = np.zeros(batch_size)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    rewards_without_first = []
    for episode in episodes:
        rewards_without_first += episode.rewards.tolist()
    rewards_without_first = np.array(rewards_without_first)

    mean = np.mean(rewards_without_first)
    std = np.std(rewards_without_first)

    scaler = StandardRewardScaler(eps=eps)
    assert not scaler.built
    scaler.fit(episodes)
    assert scaler.built

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - mean) / (std + eps)
    assert np.allclose(y, ref_y, atol=1e-4)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_return_based_reward_scaler_with_episode(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards = np.random.random(batch_size).astype("f4")
    terminals = np.zeros(batch_size)
    terminals[batch_size // 2] = 1.0
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    returns = []
    for episode in episodes:
        returns.append(episode.compute_return())

    scaler = ReturnBasedRewardScaler()
    assert not scaler.built
    scaler.fit(episodes)
    assert scaler.built

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = x.numpy() / (max(returns) - min(returns))
    assert np.allclose(y, ref_y, atol=1e-4)

    # check serialization and deserialization
    new_scaler = ReturnBasedRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.return_max == scaler.return_max
    assert new_scaler.return_min == scaler.return_min
    assert new_scaler.multiplier == scaler.multiplier


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("shift", [-1])
def test_constant_shift_reward_scaler(batch_size: int, shift: float) -> None:
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = ConstantShiftRewardScaler(shift)
    assert scaler.built

    # check trnsform
    y = scaler.transform(torch.tensor(rewards))
    assert np.allclose(y.numpy(), rewards + shift)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards)

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, rewards + shift, atol=1e-4)

    assert scaler.get_type() == "shift"

    # check serialization and deserialization
    new_scaler = ConstantShiftRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.shift == scaler.shift
