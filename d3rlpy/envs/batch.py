# pylint: disable=arguments-differ

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import gym
from gym.spaces import Discrete

from ..online.utility import get_action_size_from_env


class BatchEnvWrapper(gym.Env):  # type: ignore

    _envs: List[gym.Env]
    _observation_shape: Sequence[int]
    _action_size: int
    _discrete_action: bool
    _prev_terminals: np.ndarray

    def __init__(self, envs: List[gym.Env]):
        self._envs = envs
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self._observation_shape = self.observation_space.shape
        self._action_size = get_action_size_from_env(envs[0])
        self._discrete_action = isinstance(self.action_space, Discrete)
        self._prev_terminals = np.ones(len(envs))

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )
        rewards = np.empty(n_envs, dtype=np.float32)
        terminals = np.empty(n_envs, dtype=np.float32)
        infos = []
        for i, action in enumerate(actions):
            if self._prev_terminals[i]:
                observation = self._envs[i].reset()
                reward, terminal, info = 0.0, 0.0, {}
            else:
                observation, reward, terminal, info = self._envs[i].step(action)
            observations[i] = observation
            rewards[i] = reward
            terminals[i] = terminal
            infos.append(info)
            self._prev_terminals[i] = terminal
        return observations, rewards, terminals, infos

    def reset(self) -> np.ndarray:
        n_envs = len(self._envs)
        is_image = len(self._observation_shape) == 3
        observations = np.empty(
            (n_envs,) + tuple(self._observation_shape),
            dtype=np.uint8 if is_image else np.float32,
        )
        for i, env in enumerate(self._envs):
            observations[i] = env.reset()
        self._prev_terminals = np.ones(len(self._envs))
        return observations

    def render(self, mode: str = "human") -> Any:
        return self._envs[0].render(mode)

    @property
    def n_envs(self) -> int:
        return len(self._envs)

    def __len__(self) -> int:
        return self.n_envs
