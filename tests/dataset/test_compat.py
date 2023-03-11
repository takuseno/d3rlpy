import numpy as np
import pytest

from d3rlpy.dataset import MDPDataset, ReplayBuffer

from ..testing_utils import create_observation


@pytest.mark.parametrize("observation_shape", [(4,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("num_episodes", [10])
def test_replay_buffer(observation_shape, action_size, length, num_episodes):
    observations = []
    actions = []
    rewards = []
    terminals = []
    for _ in range(num_episodes):
        for i in range(length):
            observations.append(create_observation(observation_shape))
            actions.append(np.random.random(action_size))
            rewards.append(np.random.random())
            terminals.append(float(i == length - 1))

    dataset = MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
    )

    assert isinstance(dataset, ReplayBuffer)
    assert len(dataset.episodes) == num_episodes
