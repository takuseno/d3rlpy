from typing import Sequence

import numpy as np
import pytest

from d3rlpy.dataset import Episode, PartialTrajectory, Signature, Transition
from d3rlpy.types import DType


@pytest.mark.parametrize("shape", [(100,)])
@pytest.mark.parametrize("dtype", [np.float32])
def test_signature(shape: Sequence[int], dtype: DType) -> None:
    signature = Signature(dtype=[dtype], shape=[shape])
    data = signature.sample()
    assert data[0].shape == shape
    assert data[0].dtype == dtype


@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
def test_transition(observation_size: int, action_size: int) -> None:
    transition = Transition(
        observation=np.random.random(observation_size).astype(np.float32),
        action=np.random.random(action_size).astype(np.float32),
        reward=np.random.random(1).astype(np.float32),
        next_observation=np.random.random(observation_size).astype(np.float32),
        next_action=np.random.random(action_size).astype(np.float32),
        rewards_to_go=np.random.random((10, 1)).astype(np.float32),
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
def test_partial_trajectory(
    data_size: int, observation_size: int, action_size: int
) -> None:
    trajectory = PartialTrajectory(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.random((data_size, action_size)),
        rewards=np.random.random((data_size, 1)).astype(np.float32),
        returns_to_go=np.random.random((data_size, 1)).astype(np.float32),
        terminals=np.zeros((data_size, 1), dtype=np.float32),
        timesteps=np.arange(data_size),
        masks=np.ones(data_size, dtype=np.float32),
        length=data_size,
    )
    assert trajectory.observation_signature.shape[0] == (observation_size,)
    assert trajectory.action_signature.shape[0] == (action_size,)
    assert trajectory.reward_signature.shape[0] == (1,)
    assert len(trajectory) == data_size
    assert trajectory.get_transition_count() == data_size - 1


@pytest.mark.parametrize("data_size", [100])
@pytest.mark.parametrize("observation_size", [4])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("terminated", [False, True])
def test_episode(
    data_size: int, observation_size: int, action_size: int, terminated: bool
) -> None:
    episode = Episode(
        observations=np.random.random((data_size, observation_size)),
        actions=np.random.random((data_size, action_size)),
        rewards=np.random.random((data_size, 1)).astype(np.float32),
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
