import numpy as np
import pytest

from skbrl.dataset import _compute_rewards, _load_images, read_csv
from skbrl.dataset import MDPDataset, Episode, Transition


@pytest.mark.parametrize('data_size', [100])
@pytest.mark.parametrize('observation_size', [4])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_episodes', [4])
def test_compute_rewards(data_size, observation_size, action_size, n_episodes):
    observations = np.random.random((data_size, observation_size))
    actions = np.random.random((data_size, action_size))
    n_steps = data_size // n_episodes
    terminals = np.array(([0] * (n_steps - 1) + [1]) * n_episodes)

    def reward_func(obs_tm1, obs_t, act_t, ter_t):
        if ter_t:
            return 100.0
        return (obs_tm1 + obs_t).sum() + act_t.sum()

    # calcualate base rewards
    ref_rewards = (observations[1:] + observations[:-1]).sum(axis=1)
    ref_rewards += actions[1:].sum(axis=1)
    # append 0.0 as the initial step
    ref_rewards = np.hstack([[0.0], ref_rewards])
    # set terminal rewards
    ref_rewards[terminals == 1.0] = 100.0
    # set 0.0 to the first steps
    ref_rewards[1:][terminals[:-1] == 1.0] = 0.0

    rewards = _compute_rewards(reward_func, observations, actions, terminals)

    assert np.all(rewards == ref_rewards)
