import pytest

from d4rl_pybullet.envs import OfflineHopperBulletEnv
from d4rl_atari.offline_env import OfflineEnv
from d3rlpy.datasets import get_dataset


@pytest.mark.parametrize(
    "env_name",
    ["cartpole", "pendulum", "hopper-bullet-mixed-v0", "breakout-mixed-v0"],
)
def test_get_dataset(env_name):
    _, env = get_dataset(env_name)
    if env_name == "cartpole":
        assert env.unwrapped.spec.id == "CartPole-v0"
    elif env_name == "pendulum":
        assert env.unwrapped.spec.id == "Pendulum-v0"
    elif env_name == "hopper-bullet-mixed-v0":
        assert isinstance(env.env, OfflineHopperBulletEnv)
    elif env_name == "breakout-mixed-v0":
        assert isinstance(env.env.env, OfflineEnv)
