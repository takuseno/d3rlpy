import pytest

from d3rlpy.datasets import get_cartpole, get_dataset, get_minari, get_pendulum


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_cartpole(dataset_type: str) -> None:
    get_cartpole(dataset_type=dataset_type)


@pytest.mark.parametrize("dataset_type", ["replay", "random"])
def test_get_pendulum(dataset_type: str) -> None:
    get_pendulum(dataset_type=dataset_type)


@pytest.mark.parametrize(
    "env_name",
    ["cartpole-random", "pendulum-random"],
)
def test_get_dataset(env_name: str) -> None:
    _, env = get_dataset(env_name)
    if env_name == "cartpole-random":
        assert env.unwrapped.spec.id == "CartPole-v1"
    elif env_name == "pendulum-random":
        assert env.unwrapped.spec.id == "Pendulum-v1"


@pytest.mark.parametrize(
    "dataset_name, env_name",
    [
        ("door-cloned-v1", "AdroitHandDoor-v1"),
        ("relocate-expert-v1", "AdroitHandRelocate-v1"),
    ],
)
def test_get_minari(dataset_name: str, env_name: str) -> None:
    _, env = get_minari(dataset_name)
    assert env.unwrapped.spec.id == env_name
