from .types import GymEnv

__all__ = ["seed_env"]


def seed_env(env: GymEnv, seed: int) -> None:
    env.reset(seed=seed)
