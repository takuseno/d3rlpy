import gym
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
    create_reward_scaler,
)


@pytest.mark.parametrize(
    "scaler_type", ["clip", "multiply", "min_max", "standard"]
)
def test_create_reward_scaler(scaler_type):
    scaler = create_reward_scaler(scaler_type)
    if scaler_type == "clip":
        assert isinstance(scaler, ClipRewardScaler)
    elif scaler_type == "multiply":
        assert isinstance(scaler, MultiplyRewardScaler)
    elif scaler_type == "min_max":
        assert isinstance(scaler, MinMaxRewardScaler)
    elif scaler_type == "standard":
        assert isinstance(scaler, StandardRewardScaler)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_multiply_reward_scaler(batch_size, multiplier):
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = MultiplyRewardScaler(multiplier)

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
    params = scaler.get_params()
    assert params["multiplier"] == multiplier


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("low", [-0.1])
@pytest.mark.parametrize("high", [0.1])
@pytest.mark.parametrize("multiplier", [10.0])
def test_clip_reward_scaler(batch_size, low, high, multiplier):
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = ClipRewardScaler(low, high, multiplier)

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
    params = scaler.get_params()
    assert params["low"] == low
    assert params["high"] == high
    assert params["multiplier"] == multiplier


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("multiplier", [10.0])
def test_min_max_reward_scaler(batch_size, multiplier):
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    maximum = rewards.max()
    minimum = rewards.min()

    scaler = MinMaxRewardScaler(
        minimum=minimum, maximum=maximum, multiplier=multiplier
    )

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
    params = scaler.get_params()
    assert np.allclose(params["minimum"], minimum)
    assert np.allclose(params["maximum"], maximum)
    assert np.allclose(params["multiplier"], multiplier)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_reward_scaler_with_episode(
    observation_shape, action_size, batch_size
):
    shape = (batch_size,) + observation_shape
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
    rewards_without_first = np.array(rewards_without_first)

    maximum = rewards_without_first.max()
    minimum = rewards_without_first.min()

    scaler = MinMaxRewardScaler()
    scaler.fit(episodes)

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y, atol=1e-4)

    params = scaler.get_params()
    assert np.allclose(params["minimum"], minimum)
    assert np.allclose(params["maximum"], maximum)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
@pytest.mark.parametrize("multiplier", [10.0])
def test_standard_reward_scaler(batch_size, eps, multiplier):
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    mean = np.mean(rewards)
    std = np.std(rewards)

    scaler = StandardRewardScaler(
        mean=mean, std=std, eps=eps, multiplier=multiplier
    )

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
    params = scaler.get_params()
    assert np.allclose(params["mean"], mean)
    assert np.allclose(params["std"], std)
    assert params["eps"] == eps
    assert params["multiplier"] == multiplier


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_reward_scaler_with_episode(
    observation_shape, action_size, batch_size, eps
):
    shape = (batch_size,) + observation_shape
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
    scaler.fit(episodes)

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - mean) / (std + eps)
    assert np.allclose(y, ref_y, atol=1e-4)

    params = scaler.get_params()
    assert np.allclose(params["mean"], mean)
    assert np.allclose(params["std"], std)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_return_based_reward_scaler_with_episode(
    observation_shape, action_size, batch_size
):
    shape = (batch_size,) + observation_shape
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
    scaler.fit(episodes)

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = x.numpy() / (max(returns) - min(returns))
    assert np.allclose(y, ref_y, atol=1e-4)

    params = scaler.get_params()
    assert np.allclose(params["return_max"], max(returns))
    assert np.allclose(params["return_min"], min(returns))


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("shift", [-1])
def test_constant_shift_reward_scaler(batch_size, shift):
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = ConstantShiftRewardScaler(shift)

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
    params = scaler.get_params()
    assert params["shift"] == shift
