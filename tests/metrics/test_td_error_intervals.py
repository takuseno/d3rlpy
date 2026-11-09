from typing import Optional

import numpy as np
import pytest

from d3rlpy.algos import DQNConfig, SACConfig
from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    Episode,
    InfiniteBuffer,
    MultiStepTransitionPicker,
    ReplayBuffer,
)
from d3rlpy.metrics import TDErrorEvaluator
from d3rlpy.preprocessing import MultiplyRewardScaler, RewardScaler
from d3rlpy.types import NDArray, Observation

from .test_evaluators import DummyAlgo


class BellmanValue(DummyAlgo):
    def predict_value(self, x: Observation, action: NDArray) -> NDArray:
        assert isinstance(x, np.ndarray)
        return x[:, 0]


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.9, 1.0])
@pytest.mark.parametrize("n_steps", [1, 3, 12])
@pytest.mark.parametrize("terminated", [False, True])
def test_exact_bellman_value_has_zero_multistep_td_error(
    gamma: float, n_steps: int, terminated: bool
) -> None:
    episodes = []
    for length in (6, 9):
        rewards = np.arange(1, length + 1, dtype=np.float32)
        values = np.zeros(length, dtype=np.float32)
        if terminated:
            bootstrap = 0.0
            transition_count = length
        else:
            # The final recorded state is a truncation boundary, not terminal.
            bootstrap = 3.0
            values[-1] = bootstrap
            transition_count = length - 1
        for index in reversed(range(transition_count)):
            bootstrap = float(rewards[index]) + gamma * bootstrap
            values[index] = bootstrap
        episodes.append(
            Episode(
                observations=values.reshape(-1, 1),
                actions=np.zeros((length, 1), dtype=np.float32),
                rewards=rewards.reshape(-1, 1),
                terminated=terminated,
            )
        )
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=MultiStepTransitionPicker(n_steps, gamma),
        action_space=ActionSpace.CONTINUOUS,
        action_size=1,
    )
    algo = BellmanValue(np.zeros((1, 1), dtype=np.float32), gamma)

    # The values satisfy the one-step Bellman equation, hence every telescoped
    # n-step residual is zero, including shorter intervals near the boundary.
    assert TDErrorEvaluator()(algo, dataset) == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("n_steps", [1, 3])
@pytest.mark.parametrize(
    "reward_scaler", [None, MultiplyRewardScaler(multiplier=2.0)]
)
def test_multistep_td_error_with_native_algorithms(
    discrete: bool,
    terminated: bool,
    n_steps: int,
    reward_scaler: Optional[RewardScaler],
) -> None:
    gamma = 0.5
    length = 6
    observations = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    rewards = np.arange(1, length + 1, dtype=np.float32).reshape(-1, 1)
    episode = Episode(
        observations=observations,
        actions=np.zeros((length, 1), dtype=np.float32),
        rewards=rewards,
        terminated=terminated,
    )
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=[episode],
        transition_picker=MultiStepTransitionPicker(n_steps, gamma),
        action_space=(
            ActionSpace.DISCRETE if discrete else ActionSpace.CONTINUOUS
        ),
        action_size=2 if discrete else 1,
    )
    algo = (
        DQNConfig(gamma=gamma, reward_scaler=reward_scaler).create()
        if discrete
        else SACConfig(gamma=gamma, reward_scaler=reward_scaler).create()
    )
    algo.build_with_dataset(dataset)

    # Reconstruct targets from original episode timesteps, independently of
    # TransitionMiniBatch and the picker's precomputed reward/interval arrays.
    residuals = []
    for start in range(episode.transition_count):
        end = min(start + n_steps, length if terminated else length - 1)
        discounted_return = sum(
            gamma ** (index - start) * float(rewards[index, 0])
            for index in range(start, end)
        )
        target = np.array([discounted_return], dtype=np.float32)
        if reward_scaler:
            target = reward_scaler.transform_numpy(target)
        if end < length:
            next_observation = observations[end : end + 1]
            next_action = algo.predict(next_observation)
            next_value = algo.predict_value(next_observation, next_action)
            target += gamma ** (end - start) * next_value
        value = algo.predict_value(
            observations[start : start + 1], episode.actions[start : start + 1]
        )
        residuals.append(float((value[0] - target[0]) ** 2))

    assert TDErrorEvaluator()(algo, dataset) == pytest.approx(
        np.mean(residuals), rel=1e-5, abs=1e-7
    )
