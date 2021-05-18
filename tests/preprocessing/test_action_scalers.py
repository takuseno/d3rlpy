import gym
import numpy as np
import pytest
import torch

from d3rlpy.dataset import Episode, MDPDataset
from d3rlpy.preprocessing import MinMaxActionScaler, create_action_scaler


@pytest.mark.parametrize("scaler_type", ["min_max"])
def test_create_action_scaler(scaler_type):
    scaler = create_action_scaler(scaler_type)
    if scaler_type == "min_max":
        assert isinstance(scaler, MinMaxActionScaler)


@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_action_scaler(action_size, batch_size):
    actions = np.random.random((batch_size, action_size)).astype("f4")

    max = actions.max(axis=0)
    min = actions.min(axis=0)

    scaler = MinMaxActionScaler(maximum=max, minimum=min)

    # check range
    y = scaler.transform(torch.tensor(actions))
    assert np.all(y.numpy() >= -1.0)
    assert np.all(y.numpy() <= 1.0)

    x = torch.rand((batch_size, action_size))
    y = scaler.transform(x)
    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))
    assert np.allclose(y.numpy(), ref_y * 2.0 - 1.0)

    assert scaler.get_type() == "min_max"
    params = scaler.get_params()
    assert np.all(params["minimum"] == min)
    assert np.all(params["maximum"] == max)
    assert torch.allclose(scaler.reverse_transform(y), x, atol=1e-6)

    # check numpy
    x = np.random.random((batch_size, action_size))
    ref_y = ((max - min) * ((x + 1.0) / 2.0)) + min
    assert np.allclose(scaler.reverse_transform_numpy(x), ref_y)


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [10])
@pytest.mark.parametrize("batch_size", [32])
def test_min_max_action_scaler_with_episode(
    observation_shape, action_size, batch_size
):
    shape = (batch_size,) + observation_shape
    observations = np.random.random(shape)
    actions = np.random.random((batch_size, action_size)).astype("f4")
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
    )

    max = actions.max(axis=0)
    min = actions.min(axis=0)

    scaler = MinMaxActionScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand((batch_size, action_size))

    y = scaler.transform(x)
    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y * 2.0 - 1.0)
    params = scaler.get_params()
    assert np.all(params["minimum"] == min)
    assert np.all(params["maximum"] == max)


def test_min_max_action_scaler_with_env():
    env = gym.make("Pendulum-v0")

    scaler = MinMaxActionScaler()
    scaler.fit_with_env(env)

    assert np.all(scaler._minimum == env.action_space.low)
    assert np.all(scaler._maximum == env.action_space.high)
