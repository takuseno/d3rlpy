import pytest
import torch
import numpy as np
import gym

from d3rlpy.dataset import MDPDataset, Episode
from d3rlpy.preprocessing import create_scaler
from d3rlpy.preprocessing import PixelScaler
from d3rlpy.preprocessing import MinMaxScaler
from d3rlpy.preprocessing import StandardScaler


@pytest.mark.parametrize("scaler_type", ["pixel", "min_max", "standard"])
def test_create_scaler(scaler_type):
    scaler = create_scaler(scaler_type)
    if scaler_type == "pixel":
        assert isinstance(scaler, PixelScaler)
    elif scaler_type == "min_max":
        assert isinstance(scaler, MinMaxScaler)
    elif scaler_type == "standard":
        assert isinstance(scaler, StandardScaler)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84)])
def test_pixel_scaler(observation_shape):
    scaler = PixelScaler()

    x = torch.randint(high=255, size=observation_shape)

    y = scaler.transform(x)

    assert torch.all(y == x.float() / 255.0)

    assert scaler.get_type() == "pixel"
    assert scaler.get_params() == {}
    assert torch.all(scaler.reverse_transform(y) == x)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_scaler(observation_shape, batch_size):
    shape = (batch_size,) + observation_shape
    observations = np.random.random(shape).astype("f4")

    max = observations.max(axis=0)
    min = observations.min(axis=0)
    scaler = MinMaxScaler(maximum=max, minimum=min)

    # check range
    y = scaler.transform(torch.tensor(observations))
    assert np.all(y.numpy() >= 0.0)
    assert np.all(y.numpy() <= 1.0)

    x = torch.rand((batch_size,) + observation_shape)
    y = scaler.transform(x)
    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))
    assert np.allclose(y.numpy(), ref_y)

    assert scaler.get_type() == "min_max"
    params = scaler.get_params()
    assert np.all(params["minimum"] == min)
    assert np.all(params["maximum"] == max)
    assert torch.allclose(scaler.reverse_transform(y), x)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_scaler_with_episode(observation_shape, batch_size):
    shape = (batch_size,) + observation_shape
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1))
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    max = observations.max(axis=0)
    min = observations.min(axis=0)

    scaler = MinMaxScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand((batch_size,) + observation_shape)

    y = scaler.transform(x)
    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


def test_min_max_scaler_with_env():
    env = gym.make("BreakoutNoFrameskip-v4")

    scaler = MinMaxScaler()
    scaler.fit_with_env(env)

    x = torch.tensor(env.reset().reshape((1,) + env.observation_space.shape))
    y = scaler.transform(x)

    assert torch.all(x / 255.0 == y)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_standard_scaler(observation_shape, batch_size):
    shape = (batch_size,) + observation_shape
    observations = np.random.random(shape).astype("f4")

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardScaler(mean=mean, std=std)

    x = torch.rand((batch_size,) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - mean.reshape((1, -1))) / std.reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)

    assert scaler.get_type() == "standard"
    params = scaler.get_params()
    assert np.all(params["mean"] == mean)
    assert np.all(params["std"] == std)
    assert torch.allclose(scaler.reverse_transform(y), x, atol=1e-6)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("batch_size", [32])
def test_standard_scaler_with_episode(observation_shape, batch_size):
    shape = (batch_size,) + observation_shape
    observations = np.random.random(shape).astype("f4")
    actions = np.random.random((batch_size, 1)).astype("f4")
    rewards = np.random.random(batch_size).astype("f4")
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand((batch_size,) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - mean.reshape((1, -1))) / std.reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y, atol=1e-6)
