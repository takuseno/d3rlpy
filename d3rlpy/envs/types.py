from typing import Any, Union

import gym
import gymnasium

__all__ = ["GymEnv"]


GymEnv = Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]]
