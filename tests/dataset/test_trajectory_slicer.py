import math
from typing import Sequence

import numpy as np
import pytest

from d3rlpy.dataset import (
    BasicTrajectorySlicer,
    Episode,
    FrameStackTrajectorySlicer,
    FrameStackTransitionPicker,
    TrajectorySlicerProtocol,
)
from d3rlpy.types import Float32NDArray, Shape

from ..testing_utils import create_episode


@pytest.mark.parametrize(
    "slicer", [BasicTrajectorySlicer(), FrameStackTrajectorySlicer(n_frames=2)]
)
@pytest.mark.parametrize(
    "end_index,size,expected",
    [
        (2, 3, [1e8, 2, 1]),
        (2, 5, [0, 0, 1e8, 2, 1]),
        (1, 3, [0, 1e8, 2]),
        (2, 2, [2, 1]),
    ],
)
def test_trajectory_slicer_returns_to_go_with_large_earlier_reward(
    slicer: TrajectorySlicerProtocol,
    end_index: int,
    size: int,
    expected: Sequence[float],
) -> None:
    episode = Episode(
        observations=np.zeros((3, 1, 2, 2), dtype=np.float32),
        actions=np.zeros((3, 1), dtype=np.float32),
        rewards=np.array([[1e8], [1], [1]], dtype=np.float32),
        terminated=True,
    )

    trajectory = slicer(episode, end_index, size)

    np.testing.assert_array_equal(
        trajectory.returns_to_go,
        np.array(expected, dtype=np.float32).reshape((-1, 1)),
    )
    assert trajectory.returns_to_go.dtype == np.float32
    assert trajectory.returns_to_go.flags.c_contiguous


@pytest.mark.parametrize(
    "slicer", [BasicTrajectorySlicer(), FrameStackTrajectorySlicer(n_frames=2)]
)
@pytest.mark.parametrize(
    "rewards",
    [
        np.full((100_000, 1), 1e-6, dtype=np.float32),
        np.array([[1], [1e8], [-1e8]], dtype=np.float32),
    ],
)
def test_trajectory_slicer_returns_to_go_with_accumulation_error(
    slicer: TrajectorySlicerProtocol,
    rewards: Float32NDArray,
) -> None:
    episode = Episode(
        observations=np.zeros((len(rewards), 1), dtype=np.float32),
        actions=np.zeros((len(rewards), 1), dtype=np.float32),
        rewards=rewards,
        terminated=True,
    )

    trajectory = slicer(episode, end_index=0, size=1)

    expected = np.float32(math.fsum(rewards[:, 0]))
    np.testing.assert_array_equal(trajectory.returns_to_go, [[expected]])
    assert trajectory.returns_to_go.dtype == np.float32
    assert trajectory.returns_to_go.flags.c_contiguous


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("size", [10])
@pytest.mark.parametrize("terminated", [True, False])
def test_basic_trajectory_slicer(
    observation_shape: Shape,
    action_size: int,
    length: int,
    size: int,
    terminated: bool,
) -> None:
    episode = create_episode(
        observation_shape, action_size, length, terminated=terminated
    )

    R = 0.0
    ref_returns_to_go_list = []
    for i in reversed(range(length)):
        R += float(episode.rewards[i])
        ref_returns_to_go_list.append(R)
    ref_returns_to_go_list.reverse()
    ref_returns_to_go = np.array(ref_returns_to_go_list).reshape((-1, 1))

    slicer = BasicTrajectorySlicer()

    for i in range(size):
        end_index = i
        traj = slicer(episode, end_index, size)

        # check shape
        if isinstance(observation_shape[0], tuple):
            for j, shape in enumerate(observation_shape):
                assert isinstance(shape, tuple)
                assert traj.observations[j].shape == (size, *shape)
        else:
            assert isinstance(traj.observations, np.ndarray)
            assert traj.observations.shape == (size, *observation_shape)
        assert traj.actions.shape == (size, action_size)
        assert traj.rewards.shape == (size, 1)
        assert traj.returns_to_go.shape == (size, 1)
        assert traj.terminals.shape == (size, 1)
        assert traj.timesteps.shape == (size,)
        assert traj.masks.shape == (size,)
        assert traj.length == size

        # check values
        pad_size = size - i - 1
        end = end_index + 1
        start = max(end - size, 0)
        if isinstance(observation_shape[0], tuple):
            for j, shape in enumerate(observation_shape):
                assert np.all(
                    traj.observations[j][pad_size:]
                    == episode.observations[j][start:end]
                )
                assert np.all(traj.observations[j][:pad_size] == 0.0)
        else:
            assert np.all(
                traj.observations[pad_size:] == episode.observations[start:end]
            )
            assert np.all(traj.observations[:pad_size] == 0.0)
        assert np.all(traj.actions[pad_size:] == episode.actions[start:end])
        assert np.all(traj.actions[:pad_size] == 0.0)
        assert np.all(traj.rewards[pad_size:] == episode.rewards[start:end])
        assert np.all(traj.rewards[:pad_size] == 0.0)
        assert np.allclose(
            traj.returns_to_go[pad_size:], ref_returns_to_go[start:end]
        )
        assert np.all(traj.returns_to_go[:pad_size] == 0.0)
        assert np.all(traj.terminals == 0.0)
        assert np.all(traj.timesteps[pad_size:] == np.arange(start, end)) + 1
        assert np.all(traj.timesteps[:pad_size] == 0.0)
        assert np.all(traj.masks[pad_size:] == 1.0)
        assert np.all(traj.masks[:pad_size] == 0.0)

    # check terminal trajectory
    traj = slicer(episode, episode.size() - 1, size)
    if terminated:
        assert traj.terminals[-1][0] == 1.0
        assert np.all(traj.terminals[:-1] == 0.0)
    else:
        assert np.all(traj.terminals == 0.0)


@pytest.mark.parametrize("observation_shape", [(8,), (3, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("size", [10])
@pytest.mark.parametrize("terminated", [True, False])
@pytest.mark.parametrize("n_frames", [1, 4])
def test_frame_stack_trajectory_slicer(
    observation_shape: Sequence[int],
    action_size: int,
    length: int,
    size: int,
    terminated: bool,
    n_frames: int,
) -> None:
    episode = create_episode(
        observation_shape, action_size, length, terminated=terminated
    )

    R = 0.0
    ref_returns_to_go_list = []
    for i in reversed(range(length)):
        R += float(episode.rewards[i])
        ref_returns_to_go_list.append(R)
    ref_returns_to_go_list.reverse()
    ref_returns_to_go = np.array(ref_returns_to_go_list).reshape((-1, 1))

    slicer = FrameStackTrajectorySlicer(n_frames)
    stacked_shape = list(observation_shape)
    stacked_shape[0] *= n_frames

    # use as reference
    picker = FrameStackTransitionPicker(n_frames)

    for i in range(size):
        end_index = i
        traj = slicer(episode, end_index, size)

        # check shape
        assert isinstance(traj.observations, np.ndarray)
        assert traj.observations.shape == (size, *stacked_shape)
        assert traj.actions.shape == (size, action_size)
        assert traj.rewards.shape == (size, 1)
        assert traj.returns_to_go.shape == (size, 1)
        assert traj.terminals.shape == (size, 1)
        assert traj.timesteps.shape == (size,)
        assert traj.masks.shape == (size,)
        assert traj.length == size

        # check values
        pad_size = size - i - 1
        end = end_index + 1
        start = max(end - size, 0)

        # create reference stacked observations
        observations = []
        for j in range(end - start):
            transition = picker(episode, j + start)
            observations.append(transition.observation)
        ref_stacked_observations = np.array(observations)

        assert np.all(traj.observations[pad_size:] == ref_stacked_observations)
        assert np.all(traj.observations[:pad_size] == 0.0)
        assert np.all(traj.actions[pad_size:] == episode.actions[start:end])
        assert np.all(traj.actions[:pad_size] == 0.0)
        assert np.all(traj.rewards[pad_size:] == episode.rewards[start:end])
        assert np.all(traj.rewards[:pad_size] == 0.0)
        assert np.allclose(
            traj.returns_to_go[pad_size:], ref_returns_to_go[start:end]
        )
        assert np.all(traj.returns_to_go[:pad_size] == 0.0)
        assert np.all(traj.terminals == 0.0)
        assert np.all(traj.timesteps[pad_size:] == np.arange(start, end)) + 1
        assert np.all(traj.timesteps[:pad_size] == 0.0)
        assert np.all(traj.masks[pad_size:] == 1.0)
        assert np.all(traj.masks[:pad_size] == 0.0)

    # check terminal trajectory
    traj = slicer(episode, episode.size() - 1, size)
    if terminated:
        assert traj.terminals[-1][0] == 1.0
        assert np.all(traj.terminals[:-1] == 0.0)
    else:
        assert np.all(traj.terminals == 0.0)
