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
    ClipRewardScaler,
    ConstantShiftRewardScaler,
    MinMaxRewardScaler,
    MultiplyRewardScaler,
    ReturnBasedRewardScaler,
    StandardRewardScaler,
)
from d3rlpy.types import Float32NDArray


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_multiply_reward_scaler(batch_size: int, multiplier: float) -> None:
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = MultiplyRewardScaler(multiplier)
    assert scaler.built
    assert scaler.get_type() == "multiply"

    # check trnsform
    y = scaler.transform(torch.tensor(rewards))
    assert np.allclose(y.numpy(), rewards * multiplier)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards)

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, rewards * multiplier)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), rewards)

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
    assert scaler.get_type() == "clip"

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() <= multiplier * 0.1)
    assert np.all(y.numpy() >= multiplier * -0.1)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), np.clip(rewards, low, high))

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.all(y == multiplier * np.clip(rewards, low, high))

    # check reverse_transform_numpy
    assert np.allclose(
        scaler.reverse_transform_numpy(y), np.clip(rewards, low, high)
    )

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
    assert scaler.get_type() == "min_max"

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() >= 0.0)
    assert np.all(y.numpy() <= multiplier)

    # check reference value
    ref_y = multiplier * (rewards - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y, atol=1e-4)

    # check reverse_transform
    ref_x = ref_y * (maximum - minimum) / multiplier + minimum
    assert np.allclose(scaler.reverse_transform(y).numpy(), ref_x, atol=1e-4)

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, ref_y, atol=1e-4)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), rewards, atol=1e-4)

    # check serialization and deserialization
    new_scaler = MinMaxRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.minimum == scaler.minimum
    assert new_scaler.maximum == scaler.maximum


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_reward_scaler_with_transition_picker(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = np.max(rewards)
    minimum = np.min(rewards)

    scaler = MinMaxRewardScaler()
    assert not scaler.built
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert scaler.minimum == minimum
    assert scaler.maximum == maximum


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_reward_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
    terminals[-1] = 1.0

    episodes = EpisodeGenerator(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )()

    maximum = np.max(rewards)
    minimum = np.min(rewards)

    scaler = MinMaxRewardScaler()
    assert not scaler.built
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.minimum is not None and scaler.maximum is not None
    assert scaler.minimum == minimum
    assert scaler.maximum == maximum


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
    assert scaler.get_type() == "standard"

    # check values
    y = scaler.transform(torch.tensor(rewards))
    ref_y = multiplier * (rewards - mean) / (std + eps)
    assert np.allclose(y.numpy(), ref_y, atol=1e-3)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards, atol=1e-3)

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, ref_y, atol=1e-4)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), rewards, atol=1e-4)

    # check serialization and deserialization
    new_scaler = StandardRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.mean == scaler.mean
    assert new_scaler.std == scaler.std
    assert new_scaler.multiplier == scaler.multiplier


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_reward_scaler_with_transition_picker(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    eps: float,
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
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

    mean = np.mean(rewards_without_first)
    std = np.std(rewards_without_first)

    scaler = StandardRewardScaler(eps=eps)
    assert not scaler.built
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.mean is not None and scaler.std is not None
    assert np.allclose(scaler.mean, mean)
    assert np.allclose(scaler.std, std)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_reward_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    eps: float,
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
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

    mean = np.mean(rewards_without_first)
    std = np.std(rewards_without_first)

    scaler = StandardRewardScaler(eps=eps)
    assert not scaler.built
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.mean is not None and scaler.std is not None
    assert np.allclose(scaler.mean, mean)
    assert np.allclose(scaler.std, std)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_return_based_reward_scaler(batch_size: int, multiplier: float) -> None:
    rewards = 10.0 * np.random.random(batch_size).astype("f4")
    returns1 = np.sum(rewards[: batch_size // 2])
    returns2 = np.sum(rewards[batch_size // 2 :])

    maximum = float(np.maximum(returns1, returns2))
    minimum = float(np.minimum(returns1, returns2))

    scaler = ReturnBasedRewardScaler(
        return_min=minimum, return_max=maximum, multiplier=multiplier
    )
    assert scaler.built
    assert scaler.get_type() == "return"

    # check transform
    y = scaler.transform(torch.tensor(rewards))
    ref_y = multiplier * (rewards / (maximum - minimum))
    assert np.allclose(y.numpy(), ref_y, atol=1e-4)

    # check reverse_transform
    assert np.allclose(scaler.reverse_transform(y).numpy(), rewards, atol=1e-4)

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, ref_y, atol=1e-4)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), rewards, atol=1e-4)

    # check serialization and deserialization
    new_scaler = ReturnBasedRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.return_min == scaler.return_min
    assert new_scaler.return_max == scaler.return_max


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_return_based_reward_scaler_with_transition_picker(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
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
    scaler.fit_with_transition_picker(episodes, BasicTransitionPicker())
    assert scaler.built
    assert scaler.return_min is not None and scaler.return_max is not None
    assert scaler.return_min == np.min(returns)
    assert scaler.return_max == np.max(returns)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_return_based_reward_scaler_with_trajectory_slicer(
    observation_shape: Sequence[int], action_size: int, batch_size: int
) -> None:
    shape = (batch_size, *observation_shape)
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size))
    rewards: Float32NDArray = np.random.random(batch_size).astype(np.float32)
    terminals: Float32NDArray = np.zeros(batch_size, dtype=np.float32)
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
    scaler.fit_with_trajectory_slicer(episodes, BasicTrajectorySlicer())
    assert scaler.built
    assert scaler.return_min is not None and scaler.return_max is not None
    assert scaler.return_min == np.min(returns)
    assert scaler.return_max == np.max(returns)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("shift", [-1])
@pytest.mark.parametrize("multiplier", [1, 2])
def test_constant_shift_reward_scaler(
    batch_size: int, shift: float, multiplier: float
) -> None:
    rewards = np.random.random(batch_size).astype(np.float32) * 2 - 1

    scaler = ConstantShiftRewardScaler(shift, multiplier)
    assert scaler.built
    assert scaler.get_type() == "shift"

    # check trnsform
    y = scaler.transform(torch.tensor(rewards))
    assert np.allclose(y.numpy(), (rewards + shift) * multiplier)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards, atol=1e-4)

    # check transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, (rewards + shift) * multiplier, atol=1e-4)

    # check reverse_transform_numpy
    assert np.allclose(scaler.reverse_transform_numpy(y), rewards, atol=1e-4)

    # check serialization and deserialization
    new_scaler = ConstantShiftRewardScaler.deserialize(scaler.serialize())
    assert new_scaler.shift == scaler.shift
