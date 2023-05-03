import numpy as np
import pytest

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    DatasetInfo,
    Episode,
    PartialTrajectory,
    Signature,
    Transition,
)


@pytest.mark.parametrize("shape", [(100,)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_signature(shape, dtype):
    signature = Signature(dtype=[dtype], shape=[shape])
    data = signature.sample()
    assert data[0].shape == shape
    assert data[0].dtype == dtype


@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
def test_transition(observation_size, action_size):
    transition = Transition(
        observation=np.random.random(observation_size).astype(np.float32),
        action=np.random.random(action_size).astype(np.float32),
        reward=np.random.random(1).astype(np.float32),
        next_observation=np.random.random(observation_size).astype(np.float32),
        terminal=0.0,
        interval=1,
    )
    assert transition.observation_signature.shape[0] == (observation_size,)
    assert transition.observation_signature.dtype[0] == np.float32
    assert transition.action_signature.shape[0] == (action_size,)
    assert transition.action_signature.dtype[0] == np.float32
    assert transition.reward_signature.shape[0] == (1,)
    assert transition.reward_signature.dtype[0] == np.float32


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
def test_partial_trajectory(data_size, observation_size, action_size):
    trajectory = PartialTrajectory(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.random((data_size, action_size)),
        rewards=np.random.random((data_size, 1)),
        returns_to_go=np.random.random((data_size, 1)),
        terminals=np.zeros((data_size, 1)),
        timesteps=np.arange(data_size),
        masks=np.ones(data_size),
        length=data_size,
    )
    assert trajectory.observation_signature.shape[0] == (observation_size,)
    assert trajectory.action_signature.shape[0] == (action_size,)
    assert trajectory.reward_signature.shape[0] == (1,)
    assert len(trajectory) == data_size


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("terminated", [False, True])
def test_episode(data_size, observation_size, action_size, terminated):
    episode = Episode(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.random((data_size, action_size)),
        rewards=np.random.random((data_size, 1)),
        terminated=terminated,
    )
    assert episode.observation_signature.shape[0] == (observation_size,)
    assert episode.action_signature.shape[0] == (action_size,)
    assert episode.reward_signature.shape[0] == (1,)
    assert episode.size() == data_size
    assert len(episode) == data_size
    assert episode.compute_return() == np.sum(episode.rewards)
    if terminated:
        assert episode.transition_count == episode.size()
    else:
        assert episode.transition_count == episode.size() - 1

    episode2 = Episode.deserialize(episode.serialize())
    assert np.all(episode2.observations == episode.observations)
    assert np.all(episode2.actions == episode.actions)
    assert np.all(episode2.rewards == episode.rewards)
    assert episode2.terminated == episode.terminated


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
def test_dataset_info(data_size, observation_size, action_size):
    # continuous action space
    episode = Episode(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.random((data_size, action_size)),
        rewards=np.random.random((data_size, 1)),
        terminated=False,
    )
    dataset_info = DatasetInfo.from_episodes([episode])
    assert dataset_info.observation_signature.shape[0] == (observation_size,)
    assert dataset_info.action_signature.shape[0] == (action_size,)
    assert dataset_info.reward_signature.shape[0] == (1,)
    assert dataset_info.action_space == ActionSpace.CONTINUOUS
    assert dataset_info.action_size == action_size

    # discrete action space
    episode = Episode(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.randint(action_size, size=(data_size, 1)),
        rewards=np.random.random((data_size, 1)),
        terminated=False,
    )
    dataset_info = DatasetInfo.from_episodes([episode])
    assert dataset_info.observation_signature.shape[0] == (observation_size,)
    assert dataset_info.action_signature.shape[0] == (1,)
    assert dataset_info.action_signature.dtype[0] == np.int64
    assert dataset_info.reward_signature.shape[0] == (1,)
    assert dataset_info.action_space == ActionSpace.DISCRETE
    assert dataset_info.action_size == action_size
