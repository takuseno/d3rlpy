from functools import reduce
from operator import mul
from typing import Any, Dict, Optional, Sequence, Tuple

import gym
import numpy as np
import pytest
from gym import spaces

from d3rlpy.dataset import Observation
from d3rlpy.metrics.utility import evaluate_qlearning_with_environment
from d3rlpy.preprocessing import ActionScaler, ObservationScaler, RewardScaler


class DummyAlgo:
    def __init__(self, A: np.ndarray):
        self.A = A

    def predict(self, x: Observation) -> np.ndarray:
        x = np.array(x)
        y = np.matmul(x.reshape(x.shape[0], -1), self.A)
        return y

    def predict_value(self, x: Observation, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def sample_action(self, x: Observation) -> np.ndarray:
        raise NotImplementedError

    @property
    def gamma(self) -> float:
        return 0.99

    @property
    def observation_scaler(self) -> Optional[ObservationScaler]:
        return None

    @property
    def action_scaler(self) -> Optional[ActionScaler]:
        return None

    @property
    def reward_scaler(self) -> Optional[RewardScaler]:
        return None


class DummyEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(
        self,
        observations: np.ndarray,
        observation_shape: Sequence[int],
        episode_length: int,
    ):
        self.episode = 0
        self.episode_length = episode_length
        self.observations = observations
        self.observation_space = spaces.Box(
            low=0, high=255, shape=observation_shape
        )

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.t += 1
        observation = self.observations[self.episode - 1, self.t]
        reward = np.mean(observation) + np.mean(action)
        done = self.t == self.episode_length
        return observation, float(reward), done, False, {}

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.t = 0
        self.episode += 1
        return self.observations[self.episode - 1, 0], {}


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("episode_length", [10])
@pytest.mark.parametrize("n_trials", [10])
def test_evaluate_on_environment(
    observation_shape: Sequence[int],
    action_size: int,
    episode_length: int,
    n_trials: int,
) -> None:
    shape = (n_trials, episode_length + 1, *observation_shape)
    if len(observation_shape) == 3:
        observations = np.random.randint(0, 255, size=shape, dtype=np.uint8)
    else:
        observations = np.random.random(shape).astype("f4")

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
        algo,
        DummyEnv(observations, observation_shape, episode_length),
        n_trials,
    )
    assert np.allclose(mean_reward, np.mean(ref_rewards))
