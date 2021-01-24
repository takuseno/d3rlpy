from typing import Any, Dict, Tuple, Union

import numpy as np
import gym

from gym.spaces import Box


class ChannelFirst(gym.Wrapper):  # type: ignore

    observation_space: Box

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        low = self.observation_space.low
        high = self.observation_space.high
        dtype = self.observation_space.dtype

        # only image observation is allowed
        assert len(shape) == 3, "Image observation environment is only allowed"

        self.observation_space = Box(
            low=np.transpose(low, [2, 0, 1]),
            high=np.transpose(high, [2, 0, 1]),
            shape=(shape[2], shape[0], shape[1]),
            dtype=dtype,
        )

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        observation, reward, terminal, info = self.env.step(action)
        # make channel first observation
        observation_T = np.transpose(observation, [2, 0, 1])
        assert observation_T.shape == self.observation_space.shape
        return observation_T, reward, terminal, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        observation = self.env.reset(**kwargs)
        # make channel first observation
        observation_T = np.transpose(observation, [2, 0, 1])
        assert observation_T.shape == self.observation_space.shape
        return observation_T
