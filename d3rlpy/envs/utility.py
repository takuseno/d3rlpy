from typing import Any

import gym

__all__ = ["seed_env"]


def seed_env(env: gym.Env[Any, Any], seed: int) -> None:
    env.reset(seed=seed)
