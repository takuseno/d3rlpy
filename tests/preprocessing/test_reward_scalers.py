import gym
import numpy as np
import pytest
import torch

from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy.preprocessing import (
    ClipRewardScaler,
    MinMaxRewardScaler,
    StandardRewardScaler,
    create_reward_scaler,
)


@pytest.mark.parametrize("scaler_type", ["clip", "min_max", "standard"])
def test_create_reward_scaler(scaler_type):
    scaler = create_reward_scaler(scaler_type)
    if scaler_type == "clip":
        assert isinstance(scaler, ClipRewardScaler)
    elif scaler_type == "min_max":
        assert isinstance(scaler, MinMaxRewardScaler)
    elif scaler_type == "standard":
        assert isinstance(scaler, StandardRewardScaler)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("low", [-0.1])
@pytest.mark.parametrize("high", [0.1])
def test_clip_reward_scaler(batch_size, low, high):
    rewards = np.random.random(batch_size).astype("f4") * 2 - 1

    scaler = ClipRewardScaler(low, high)

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() <= 0.1)
    assert np.all(y.numpy() >= -0.1)

    # check reverse_transform
    y = scaler.reverse_transform(torch.tensor(rewards))
    assert np.allclose(y.numpy(), rewards)

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.all(y == np.clip(rewards, low, high))

    assert scaler.get_type() == "clip"
    params = scaler.get_params()
    assert params["low"] == low
    assert params["high"] == high


@pytest.mark.parametrize("batch_size", [32])
def test_min_max_reward_scaler(batch_size):
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    maximum = rewards.max()
    minimum = rewards.min()

    scaler = MinMaxRewardScaler(minimum=minimum, maximum=maximum)

    # check range
    y = scaler.transform(torch.tensor(rewards))
    assert np.all(y.numpy() >= 0.0)
    assert np.all(y.numpy() <= 1.0)

    # check reference value
    ref_y = (rewards - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y)

    # check reverse_transform
    ref_x = ref_y * (maximum - minimum) + minimum
    assert np.allclose(scaler.reverse_transform(y).numpy(), ref_x)

    # check reverse_transform_numpy
    assert np.allclose(scaler.transform_numpy(rewards), ref_y)

    assert scaler.get_type() == "min_max"
    params = scaler.get_params()
    assert np.allclose(params["minimum"], minimum)
    assert np.allclose(params["maximum"], maximum)


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
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    maximum = rewards.max()
    minimum = rewards.min()

    scaler = MinMaxRewardScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - minimum) / (maximum - minimum)
    assert np.allclose(y.numpy(), ref_y)

    params = scaler.get_params()
    assert np.allclose(params["minimum"], minimum)
    assert np.allclose(params["maximum"], maximum)


@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("eps", [0.3])
def test_standard_reward_scaler(batch_size, eps):
    rewards = 10.0 * np.random.random(batch_size).astype("f4")

    mean = np.mean(rewards)
    std = np.std(rewards)

    scaler = StandardRewardScaler(mean=mean, std=std, eps=eps)

    # check values
    y = scaler.transform(torch.tensor(rewards))
    ref_y = (rewards - mean) / (std + eps)
    assert np.allclose(y.numpy(), ref_y)

    # check reverse_transform
    x = scaler.reverse_transform(y)
    assert np.allclose(x.numpy(), rewards)

    # check reverse_transform_numpy
    y = scaler.transform_numpy(rewards)
    assert np.allclose(y, ref_y)

    assert scaler.get_type() == "standard"
    params = scaler.get_params()
    assert np.allclose(params["mean"], mean)
    assert np.allclose(params["std"], std)
    assert params["eps"] == eps


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
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    mean = np.mean(rewards)
    std = np.std(rewards)

    scaler = StandardRewardScaler(eps=eps)
    scaler.fit(dataset.episodes)

    x = torch.rand(batch_size)
    y = scaler.transform(x)
    ref_y = (x.numpy() - mean) / (std + eps)
    assert np.allclose(y, ref_y, atol=1e-6)

    params = scaler.get_params()
    assert np.allclose(params["mean"], mean)
    assert np.allclose(params["std"], std)
