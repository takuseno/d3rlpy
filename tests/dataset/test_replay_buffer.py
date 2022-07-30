import os

import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    EpisodeGenerator,
    ExperienceWriter,
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
    replay_buffer = ReplayBuffer(InfiniteBuffer())

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
    replay_buffer = ReplayBuffer(InfiniteBuffer(), [episode])

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
def test_replay_buffer_sample(
    observation_shape, action_size, length, partial_length, batch_size
):
    episode = create_episode(observation_shape, action_size, length)
    replay_buffer = ReplayBuffer(InfiniteBuffer(), [episode])

    # check transition sampling
    picker = BasicTransitionPicker()
    batch = replay_buffer.sample_transition_batch(picker, batch_size)
    assert len(batch) == batch_size

    # check trajectory sampling
    slicer = BasicTrajectorySlicer()
    batch = replay_buffer.sample_trajectory_batch(
        slicer, batch_size, partial_length
    )
    assert len(batch) == batch_size


@pytest.mark.parametrize("limit", [100])
def test_create_fifo_replay_buffer(limit):
    replay_buffer = create_fifo_replay_buffer(limit)
    assert isinstance(replay_buffer.buffer, FIFOBuffer)


def test_create_infinite_replay_buffer():
    replay_buffer = create_infinite_replay_buffer()
    assert isinstance(replay_buffer.buffer, InfiniteBuffer)
