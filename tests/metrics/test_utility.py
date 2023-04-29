from functools import reduce
from operator import mul

import numpy as np
import pytest
from gym import spaces

from d3rlpy.metrics.utility import evaluate_qlearning_with_environment


class DummyAlgo:
    def __init__(self, A):
        self.A = A

    def predict(self, x):
        x = np.array(x)
        y = np.matmul(x.reshape(x.shape[0], -1), self.A)
        return y


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("n_trials", [10])
def test_evaluate_on_environment(
    observation_shape, action_size, episode_length, n_trials
):
    shape = (n_trials, episode_length + 1) + observation_shape
    if len(observation_shape) == 3:
        observations = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    else:
        observations = np.random.random(shape).astype("f4")

    class DummyEnv:
        def __init__(self):
            self.episode = 0
            self.observation_space = spaces.Box(
                low=0, high=255, shape=observation_shape
            )

        def step(self, action):
            self.t += 1
            observation = observations[self.episode - 1, self.t]
            reward = np.mean(observation) + np.mean(action)
            done = self.t == episode_length
            return observation, reward, done, False, {}

        def reset(self):
            self.t = 0
            self.episode += 1
            return observations[self.episode - 1, 0], {}

    # projection matrix for deterministic action
    feature_size = reduce(mul, observation_shape)
    A = np.random.random((feature_size, action_size))
    algo = DummyAlgo(A)

    ref_rewards = []
    for i in range(n_trials):
        episode_obs = observations[i].reshape((-1, feature_size))
        actions = algo.predict(episode_obs[:-1])
        rewards = np.mean(episode_obs[1:], axis=1) + np.mean(actions, axis=1)
        ref_rewards.append(np.sum(rewards))

    mean_reward = evaluate_qlearning_with_environment(
        algo, DummyEnv(), n_trials
    )
    assert np.allclose(mean_reward, np.mean(ref_rewards))
