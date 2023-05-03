import numpy as np
import pytest

from d3rlpy.dataset import BasicTransitionPicker, FrameStackTransitionPicker

from ..testing_utils import create_episode


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
def test_basic_transition_picker(observation_shape, action_size, length):
    episode = create_episode(
        observation_shape, action_size, length, terminated=True
    )

    picker = BasicTransitionPicker()

    # check transition
    transition = picker(episode, 0)
    if isinstance(observation_shape[0], tuple):
        for i, shape in enumerate(observation_shape):
            assert transition.observation_signature.shape[i] == shape
            assert np.all(
                transition.observation[i] == episode.observations[i][0]
            )
            assert np.all(
                transition.next_observation[i] == episode.observations[i][1]
            )
    else:
        assert transition.observation_signature.shape[0] == observation_shape
        assert np.all(transition.observation == episode.observations[0])
        assert np.all(transition.next_observation == episode.observations[1])
    assert np.all(transition.action == episode.actions[0])
    assert np.all(transition.reward == episode.rewards[0])
    assert transition.interval == 1
    assert transition.terminal == 0

    # check terminal transition
    transition = picker(episode, length - 1)
    if isinstance(observation_shape[0], tuple):
        for i, shape in enumerate(observation_shape):
            dummy_observation = np.zeros(shape)
            assert transition.observation_signature.shape[i] == shape
            assert np.all(
                transition.observation[i] == episode.observations[i][-1]
            )
            assert np.all(transition.next_observation[i] == dummy_observation)
    else:
        dummy_observation = np.zeros(observation_shape)
        assert transition.observation_signature.shape[0] == observation_shape
        assert np.all(transition.observation == episode.observations[-1])
        assert np.all(transition.next_observation == dummy_observation)
    assert np.all(transition.action == episode.actions[-1])
    assert np.all(transition.reward == episode.rewards[-1])
    assert transition.interval == 1
    assert transition.terminal == 1.0


@pytest.mark.parametrize("observation_shape", [(8,), (3, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("n_frames", [4])
def test_frame_stack_transition_picker(
    observation_shape, action_size, length, n_frames
):
    episode = create_episode(
        observation_shape, action_size, length, terminated=True
    )

    picker = FrameStackTransitionPicker(n_frames)

    n_channels = observation_shape[0]
    ref_observation_shape = (n_frames * n_channels, *observation_shape[1:])

    # check stacked frames
    for i in range(n_frames):
        transition = picker(episode, i)
        assert (
            transition.observation_signature.shape[0] == ref_observation_shape
        )
        for j in range(n_frames):
            obs = transition.observation[j * n_channels : (j + 1) * n_channels]
            if j >= n_frames - i - 1:
                index = j - n_frames + i + 1
                assert np.all(obs == episode.observations[index])
            else:
                assert np.all(obs == 0.0)

            next_obs = transition.next_observation[
                j * n_channels : (j + 1) * n_channels
            ]
            if j + 1 >= n_frames - i - 1:
                index = j - n_frames + i + 2
                assert np.all(next_obs == episode.observations[index])
            else:
                assert np.all(next_obs == 0.0)
        assert np.all(transition.action == episode.actions[i])
        assert np.all(transition.reward == episode.rewards[i])
        assert transition.terminal == 0.0
        assert transition.interval == 1

    # check terminal state
    transition = picker(episode, length - 1)
    assert np.all(transition.next_observation == 0.0)
    assert transition.terminal == 1.0
