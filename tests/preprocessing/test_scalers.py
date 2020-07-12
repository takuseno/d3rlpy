import pytest
import torch
import numpy as np

from d3rlpy.dataset import MDPDataset, Episode
from d3rlpy.preprocessing import PixelScaler
from d3rlpy.preprocessing import MinMaxScaler
from d3rlpy.preprocessing import StandardScaler


@pytest.mark.parametrize('observation_shape', [(4, 84, 84)])
def test_pixel_scaler(observation_shape):
    scaler = PixelScaler()

    x = torch.randint(high=255, size=observation_shape)

    y = scaler.transform(x)

    assert torch.all(y == x.float() / 255.0)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_min_max_scaler(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)

    max = observations.max(axis=0)
    min = observations.min(axis=0)

    scaler = MinMaxScaler(maximum=max, minimum=min)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_min_max_scaler_with_dataset(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)
    actions = np.random.random((batch_size, 1))
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    max = observations.max(axis=0)
    min = observations.min(axis=0)

    scaler = MinMaxScaler(dataset)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_min_max_scaler_with_episode(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)
    actions = np.random.random((batch_size, 1))
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(observations, actions, rewards, terminals)

    max = observations.max(axis=0)
    min = observations.min(axis=0)

    scaler = MinMaxScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - min.reshape((1, -1))) / (max - min).reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_standard_scaler(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardScaler(mean=mean, std=std)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - mean.reshape((1, -1))) / std.reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_standard_scaler_with_dataset(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)
    actions = np.random.random((batch_size, 1))
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardScaler(dataset)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - mean.reshape((1, -1))) / std.reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('batch_size', [32])
def test_standard_scaler_with_episode(observation_shape, batch_size):
    observations = np.random.random((batch_size, ) + observation_shape)
    actions = np.random.random((batch_size, 1))
    rewards = np.random.random(batch_size)
    terminals = np.random.randint(2, size=batch_size)
    terminals[-1] = 1.0

    dataset = MDPDataset(observations, actions, rewards, terminals)

    mean = observations.mean(axis=0)
    std = observations.std(axis=0)

    scaler = StandardScaler()
    scaler.fit(dataset.episodes)

    x = torch.rand((batch_size, ) + observation_shape)

    y = scaler.transform(x)

    ref_y = (x.numpy() - mean.reshape((1, -1))) / std.reshape((1, -1))

    assert np.allclose(y.numpy(), ref_y)
