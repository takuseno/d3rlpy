from collections import OrderedDict
from typing import Any

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces

from d3rlpy.envs.wrappers import GoalConcatWrapper
from d3rlpy.types import NDArray


class GoalEnv(gymnasium.Env[dict[str, Any], Any]):
    def __init__(self, dict_goal: bool, dtype: Any):
        goal_spaces: dict[str, spaces.Space[Any]] = OrderedDict(
            z=spaces.Box(10, 20, shape=(2,), dtype=dtype),
            a=spaces.Box(-5, -1, shape=(1,), dtype=dtype),
        )
        goal_space = (
            spaces.Dict(goal_spaces)
            if dict_goal
            else spaces.Box(-5, 20, shape=(3,), dtype=dtype)
        )
        self.observation_space = spaces.Dict(
            observation=spaces.Box(-1, 1, shape=(2,), dtype=dtype),
            desired_goal=goal_space,
        )
        self.action_space = spaces.Discrete(2)

    def reset(
        self, *, seed: Any = None, options: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return self.observation_space.sample(), 1.0, False, False, {}


@pytest.mark.parametrize("dict_goal", [True, False])
@pytest.mark.parametrize("tuple_observation", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_goal_concat_observations_match_space(
    dict_goal: bool, tuple_observation: bool, dtype: Any
) -> None:
    raw_env = GoalEnv(dict_goal, dtype)
    env = GoalConcatWrapper(raw_env, tuple_observation=tuple_observation)
    observation, _ = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    observation, reward, terminated, truncated, _ = env.step(0)
    assert env.observation_space.contains(observation)
    assert (reward, terminated, truncated) == (1.0, False, False)
    if tuple_observation:
        assert isinstance(env.observation_space, spaces.Tuple)
        goal_space = env.observation_space[1]
        assert isinstance(goal_space, spaces.Box)
        assert isinstance(observation, tuple)
        goal = observation[1]
    else:
        assert isinstance(env.observation_space, spaces.Box)
        assert isinstance(observation, np.ndarray)
        goal_space = env.observation_space
        goal = observation[2:]
    assert goal.dtype == dtype
    assert goal.shape == (3,)
    if dict_goal:
        expected_low: NDArray = np.array([-5, 10, 10], dtype=dtype)
        expected_high: NDArray = np.array([-1, 20, 20], dtype=dtype)
        np.testing.assert_array_equal(goal_space.low[-3:], expected_low)
        np.testing.assert_array_equal(goal_space.high[-3:], expected_high)
        assert isinstance(raw_env.observation_space, spaces.Dict)
        assert isinstance(
            raw_env.observation_space["desired_goal"], spaces.Dict
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_goal_concat_in_vector_env(dtype: Any) -> None:
    env = gymnasium.vector.SyncVectorEnv(
        [
            lambda: GoalConcatWrapper(
                GoalEnv(True, dtype), tuple_observation=True
            )
            for _ in range(2)
        ]
    )
    observation: Any
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    observation, _, _, _, _ = env.step(np.zeros(2, dtype=np.int64))
    assert env.observation_space.contains(observation)
    env.close()
