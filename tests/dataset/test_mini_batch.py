import numpy as np
import pytest

from d3rlpy.dataset import TrajectoryMiniBatch, TransitionMiniBatch
from d3rlpy.types import Shape

from ..testing_utils import create_partial_trajectory, create_transition


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
def test_transition_mini_batch(
    observation_shape: Shape, action_size: int, batch_size: int
) -> None:
    transitions = []
    for _ in range(batch_size):
        transition = create_transition(
            observation_shape,
            action_size,
            terminated=bool(np.random.randint(2)),
        )
        transitions.append(transition)

    batch = TransitionMiniBatch.from_transitions(transitions)

    ref_actions = np.array([t.action for t in transitions])
    ref_next_actions = np.array([t.next_action for t in transitions])
    ref_rewards = np.array([t.reward for t in transitions])
    ref_terminals = np.array([[t.terminal] for t in transitions])
    ref_intervals = np.array([[t.interval] for t in transitions])

    if isinstance(observation_shape[0], tuple):
        for i, shape in enumerate(observation_shape):
            ref_observations = np.array([t.observation[i] for t in transitions])
            ref_next_observations = np.array(
                [t.next_observation[i] for t in transitions]
            )
            assert isinstance(shape, tuple)
            assert batch.observations[i].shape == (batch_size, *shape)
            assert batch.next_observations[i].shape == (batch_size, *shape)
            assert np.all(batch.observations[i] == ref_observations)
            assert np.all(batch.next_observations[i] == ref_next_observations)
    else:
        ref_observations = np.array([t.observation for t in transitions])
        ref_next_observations = np.array(
            [t.next_observation for t in transitions]
        )
        assert isinstance(batch.observations, np.ndarray)
        assert isinstance(batch.next_observations, np.ndarray)
        assert batch.observations.shape == (batch_size, *observation_shape)
        assert batch.next_observations.shape == (batch_size, *observation_shape)
        assert np.all(batch.observations == ref_observations)
        assert np.all(batch.next_observations == ref_next_observations)
    assert batch.actions.shape == (batch_size, action_size)
    assert batch.next_actions.shape == (batch_size, action_size)
    assert batch.rewards.shape == (batch_size, 1)
    assert batch.terminals.shape == (batch_size, 1)
    assert batch.intervals.shape == (batch_size, 1)
    assert np.all(batch.actions == ref_actions)
    assert np.all(batch.next_actions == ref_next_actions)
    assert np.all(batch.rewards == ref_rewards)
    assert np.all(batch.terminals == ref_terminals)
    assert np.all(batch.intervals == ref_intervals)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("length", [100])
def test_trajectory_mini_batch(
    observation_shape: Shape,
    action_size: int,
    batch_size: int,
    length: int,
) -> None:
    trajectories = []
    for _ in range(batch_size):
        traj = create_partial_trajectory(observation_shape, action_size, length)
        trajectories.append(traj)

    batch = TrajectoryMiniBatch.from_partial_trajectories(trajectories)

    if isinstance(observation_shape[0], tuple):
        for i, shape in enumerate(observation_shape):
            assert isinstance(shape, tuple)
            assert batch.observations[i].shape == (batch_size, length, *shape)
    else:
        assert isinstance(batch.observations, np.ndarray)
        assert batch.observations.shape == (
            batch_size,
            length,
            *observation_shape,
        )
    assert batch.actions.shape == (batch_size, length, action_size)  # type: ignore
    assert batch.rewards.shape == (batch_size, length, 1)  # type: ignore
    assert batch.returns_to_go.shape == (batch_size, length, 1)  # type: ignore
    assert batch.terminals.shape == (batch_size, length, 1)  # type: ignore
    assert batch.timesteps.shape == (batch_size, length)
    assert batch.masks.shape == (batch_size, length)
    assert batch.length == length
