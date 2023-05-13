import numpy as np
import pytest

from d3rlpy.dataset import BasicTrajectorySlicer, Shape

from ..testing_utils import create_episode


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
        assert np.all(traj.terminals == 0.0)
        assert np.all(traj.timesteps[pad_size:] == np.arange(start, end))
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
