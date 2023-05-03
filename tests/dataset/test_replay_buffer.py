import os

import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    FIFOBuffer,
    InfiniteBuffer,
    ReplayBuffer,
    create_fifo_replay_buffer,
    create_infinite_replay_buffer,
)

from ..testing_utils import create_episode, create_observation


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("terminated", [False, True])
def test_replay_buffer(observation_shape, action_size, length, terminated):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = ReplayBuffer(
        InfiniteBuffer(),
        observation_signature=episode.observation_signature,
        action_signature=episode.action_signature,
        reward_signature=episode.reward_signature,
    )

    for _ in range(length):
        replay_buffer.append(
            observation=create_observation(observation_shape),
            action=np.random.random(action_size),
            reward=np.random.random(),
        )
    replay_buffer.clip_episode(terminated)

    if terminated:
        assert replay_buffer.transition_count == length
    else:
        assert replay_buffer.transition_count == length - 1


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
def test_replay_buffer_dump_load(observation_shape, action_size, length):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = ReplayBuffer(InfiniteBuffer(), episodes=[episode])

    # save
    with open(os.path.join("test_data", "replay_buffer.h5"), "w+b") as f:
        replay_buffer.dump(f)

    # load
    with open(os.path.join("test_data", "replay_buffer.h5"), "rb") as f:
        replay_buffer2 = ReplayBuffer.load(f, InfiniteBuffer())
    assert replay_buffer2.transition_count == replay_buffer.transition_count


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("partial_length", [10])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("picker", [None, BasicTransitionPicker()])
@pytest.mark.parametrize("slicer", [None, BasicTrajectorySlicer()])
def test_replay_buffer_sample(
    observation_shape,
    action_size,
    length,
    partial_length,
    batch_size,
    picker,
    slicer,
):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = ReplayBuffer(
        InfiniteBuffer(),
        episodes=[episode],
        transition_picker=picker,
        trajectory_slicer=slicer,
    )

    # check transition sampling
    batch = replay_buffer.sample_transition_batch(batch_size)
    assert len(batch) == batch_size

    # check trajectory sampling
    batch = replay_buffer.sample_trajectory_batch(batch_size, partial_length)
    assert len(batch) == batch_size


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("limit", [100])
def test_create_fifo_replay_buffer(
    observation_shape, action_size, length, limit
):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = create_fifo_replay_buffer(limit, episodes=[episode])
    assert isinstance(replay_buffer.buffer, FIFOBuffer)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
def test_create_infinite_replay_buffer(observation_shape, action_size, length):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = create_infinite_replay_buffer(episodes=[episode])
    assert isinstance(replay_buffer.buffer, InfiniteBuffer)
