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
        ("door-cloned-v2", "AdroitHandDoor-v1"),
        ("relocate-expert-v2", "AdroitHandRelocate-v1"),
        ("kitchen-complete-v1", "FrankaKitchen-v1"),
    ],
)
@pytest.mark.parametrize("tuple_observation", [False, True])
def test_get_minari(
    dataset_name: str, env_name: str, tuple_observation: bool
) -> None:
    dataset, env = get_minari(dataset_name, tuple_observation=tuple_observation)
    assert env.unwrapped.spec.id == env_name  # type: ignore

    if tuple_observation:
        # check shape
        ep = dataset.episodes[0]
        ref_shape0 = ep.observations[0].shape[1:]
        ref_shape1 = ep.observations[1].shape[1:]
        obs, _ = env.reset()
        assert obs[0].shape == ref_shape0
        assert obs[1].shape == ref_shape1
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs[0].shape == ref_shape0
        assert obs[1].shape == ref_shape1
    else:
        # check shape
        ref_shape = dataset.episodes[0].observations.shape[1:]  # type: ignore
        obs, _ = env.reset()
        assert obs.shape == ref_shape
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == ref_shape
