from typing import Any, Dict, Tuple, Union

import numpy as np
import gym

from gym.spaces import Box
from gym.wrappers import AtariPreprocessing, TransformReward


class ChannelFirst(gym.Wrapper):  # type: ignore
    """Channel-first wrapper for image observation environments.

    d3rlpy expects channel-first images since it's built with PyTorch.
    You can transform the observation shape with ``ChannelFirst`` wrapper.

    Args:
        env (gym.Env): gym environment.

    """

    observation_space: Box

    def __init__(self, env: gym.Env):
        super().__init__(env)
        shape = self.observation_space.shape
        low = self.observation_space.low
        high = self.observation_space.high
        dtype = self.observation_space.dtype

        if len(shape) == 3:
            self.observation_space = Box(
                low=np.transpose(low, [2, 0, 1]),
                high=np.transpose(high, [2, 0, 1]),
                shape=(shape[2], shape[0], shape[1]),
                dtype=dtype,
            )
        elif len(shape) == 2:
            self.observation_space = Box(
                low=np.reshape(low, (1, *shape)),
                high=np.reshape(high, (1, *shape)),
                shape=(1, *shape),
                dtype=dtype,
            )
        else:
            raise ValueError("image observation is only allowed.")

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
        observation, reward, terminal, info = self.env.step(action)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T, reward, terminal, info

    def reset(self, **kwargs: Any) -> np.ndarray:
        observation = self.env.reset(**kwargs)
        # make channel first observation
        if observation.ndim == 3:
            observation_T = np.transpose(observation, [2, 0, 1])
        else:
            observation_T = np.reshape(observation, (1, *observation.shape))
        assert observation_T.shape == self.observation_space.shape
        return observation_T


class Atari(gym.Wrapper):  # type: ignore
    """Atari 2600 wrapper for experiments.

    Args:
        env (gym.Env): gym environment.
        is_eval (bool): flag to enter evaluation mode.

    """

    def __init__(self, env: gym.Env, is_eval: bool = False):
        env = AtariPreprocessing(env, terminal_on_life_loss=not is_eval)
        if not is_eval:
            env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))
        super().__init__(ChannelFirst(env))
