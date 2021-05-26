import pytest
from d4rl_pybullet.envs import OfflineHopperBulletEnv

from d3rlpy.datasets import get_cartpole, get_dataset, get_pendulum


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_cartpole(dataset_type):
    get_cartpole(dataset_type=dataset_type)


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_pendulum(dataset_type):
    get_pendulum(dataset_type=dataset_type)


@pytest.mark.parametrize(
    "env_name",
    ["cartpole", "pendulum", "hopper-bullet-mixed-v0"],
)
def test_get_dataset(env_name):
    _, env = get_dataset(env_name)
    if env_name == "cartpole":
        assert env.unwrapped.spec.id == "CartPole-v0"
    elif env_name == "pendulum":
        assert env.unwrapped.spec.id == "Pendulum-v0"
    elif env_name == "hopper-bullet-mixed-v0":
        assert isinstance(env.env, OfflineHopperBulletEnv)
