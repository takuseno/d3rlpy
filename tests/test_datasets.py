import pytest

from d3rlpy.datasets import get_cartpole, get_dataset, get_pendulum


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_cartpole(dataset_type):
    get_cartpole(dataset_type=dataset_type)


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_pendulum(dataset_type):
    get_pendulum(dataset_type=dataset_type)


@pytest.mark.parametrize(
    "env_name",
    ["cartpole-random", "pendulum-random"],
)
def test_get_dataset(env_name):
    _, env = get_dataset(env_name)
    if env_name == "cartpole-random":
        assert env.unwrapped.spec.id == "CartPole-v1"
    elif env_name == "pendulum-random":
        assert env.unwrapped.spec.id == "Pendulum-v1"
