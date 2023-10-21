from typing import Any, Sequence, Tuple, Union, cast

import gym
import gymnasium
import numpy as np
import pytest

from d3rlpy.constants import ActionSpace
from d3rlpy.dataset import (
    batch_pad_array,
    batch_pad_observations,
    cast_recursively,
    check_dtype,
    check_non_1d_array,
    create_zero_observation,
    detect_action_size_from_env,
    detect_action_space,
    detect_action_space_from_env,
    get_dtype_from_observation,
    get_dtype_from_observation_sequence,
    get_shape_from_observation,
    get_shape_from_observation_sequence,
    retrieve_observation,
    slice_observations,
    stack_observations,
    stack_recent_observations,
)
from d3rlpy.types import DType, Shape

from ..testing_utils import create_observation, create_observations


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("index", [0])
def test_retrieve_observation(
    observation_shape: Shape, length: int, index: int
) -> None:
    observations = create_observations(observation_shape, length)

    observation = retrieve_observation(observations, index)

    if isinstance(observation, list):
        for i, obs in enumerate(observation):
            assert np.all(obs == observations[i][index])
    else:
        assert np.all(observation == observations[index])


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
def test_create_zero_observation(observation_shape: Shape) -> None:
    observation = create_observation(observation_shape)

    zero_observation = create_zero_observation(observation)

    if isinstance(zero_observation, list):
        for i, obs in enumerate(zero_observation):
            assert obs.shape == observation_shape[i]
            assert np.all(obs == 0.0)
    else:
        assert isinstance(zero_observation, np.ndarray)
        assert zero_observation.shape == observation_shape
        assert np.all(zero_observation == 0.0)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("index", [(0, 5)])
def test_slice_observations(
    observation_shape: Shape, length: int, index: Tuple[int, int]
) -> None:
    observations = create_observations(observation_shape, length)

    start, end = index
    size = end - start
    sliced_observations = slice_observations(observations, start, end)

    if isinstance(sliced_observations, list):
        for i, shape in enumerate(observation_shape):
            assert isinstance(shape, tuple)
            assert sliced_observations[i].shape == (size, *shape)
            assert np.all(sliced_observations[i] == observations[i][start:end])
    else:
        assert isinstance(sliced_observations, np.ndarray)
        assert sliced_observations.shape == (size, *observation_shape)
        assert np.all(sliced_observations == observations[start:end])


@pytest.mark.parametrize(
    "shape",
    [
        (
            10,
            4,
        ),
        (10, 3, 84, 84),
    ],
)
@pytest.mark.parametrize("pad_size", [5])
def test_batch_pad_array(shape: Sequence[int], pad_size: int) -> None:
    array = np.random.random(shape)

    padded_array = batch_pad_array(array, pad_size)

    assert padded_array.shape == (pad_size + shape[0], *shape[1:])
    assert np.all(padded_array[pad_size:] == array)
    assert np.all(padded_array[:pad_size] == 0.0)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,))])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("pad_size", [5])
def test_batch_pad_observations(
    observation_shape: Shape, length: int, pad_size: int
) -> None:
    observations = create_observations(observation_shape, length)

    padded_observations = batch_pad_observations(observations, pad_size)

    if isinstance(padded_observations, list):
        for i, shape in enumerate(observation_shape):
            assert isinstance(shape, tuple)
            assert padded_observations[i].shape == (pad_size + length, *shape)
            assert np.all(padded_observations[i][pad_size:] == observations[i])
            assert np.all(padded_observations[i][:pad_size] == 0.0)
    else:
        assert isinstance(padded_observations, np.ndarray)
        assert padded_observations.shape == (
            pad_size + length,
            *observation_shape,
        )
        assert np.all(padded_observations[pad_size:] == observations)
        assert np.all(padded_observations[:pad_size] == 0.0)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("n_frames", [2])
def test_stack_recent_observations(
    observation_shape: Shape, length: int, index: int, n_frames: int
) -> None:
    observations = create_observations(observation_shape, length)

    stacked_observation = stack_recent_observations(
        observations, index, n_frames
    )

    if isinstance(stacked_observation, list):
        for i, shape in enumerate(observation_shape):
            assert isinstance(shape, tuple)
            assert stacked_observation[i].shape == (
                n_frames * shape[0],
                *shape[1:],
            )
            pad_size = (n_frames - index - 1) * shape[0]
            if pad_size > 0:
                assert np.all(stacked_observation[i][:pad_size] == 0.0)
                ref_obs = observations[i][: index + 1].reshape((-1, *shape[1:]))
                assert np.all(stacked_observation[i][pad_size:] == ref_obs)
            else:
                ref_obs = observations[i][: index + 1].reshape((-1, *shape[1:]))
                assert np.all(stacked_observation[i] == ref_obs)
    else:
        assert isinstance(observations, np.ndarray)
        assert isinstance(stacked_observation, np.ndarray)
        assert isinstance(observation_shape[0], int)
        assert stacked_observation.shape == (
            n_frames * observation_shape[0],
            *observation_shape[1:],
        )
        pad_size = (n_frames - index - 1) * observation_shape[0]
        rest_shape = cast(Sequence[int], observation_shape[1:])
        if pad_size > 0:
            assert np.all(stacked_observation[:pad_size] == 0.0)
            ref_obs = observations[: index + 1].reshape((-1, *rest_shape))
            assert np.all(stacked_observation[pad_size:] == ref_obs)
        else:
            ref_obs = observations[: index + 1].reshape((-1, *rest_shape))
            assert np.all(stacked_observation == ref_obs)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
@pytest.mark.parametrize("size", [100])
def test_stack_observations(observation_shape: Shape, size: int) -> None:
    observations = [create_observation(observation_shape) for _ in range(size)]

    stacked_observations = stack_observations(observations)

    if isinstance(stacked_observations, list):
        ref_obs_list = [
            np.stack([observations[j][i] for j in range(size)])
            for i in range(len(observation_shape))
        ]
        for i, shape in enumerate(observation_shape):
            assert isinstance(shape, tuple)
            assert stacked_observations[i].shape == (size, *shape)
            assert np.all(stacked_observations[i] == ref_obs_list[i])
    else:
        assert isinstance(stacked_observations, np.ndarray)
        assert stacked_observations.shape == (size, *observation_shape)
        ref_obs = np.stack(observations)
        assert np.all(stacked_observations == ref_obs)


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
def test_get_shape_from_observation(observation_shape: Shape) -> None:
    observation = create_observation(observation_shape)
    assert tuple(get_shape_from_observation(observation)) == observation_shape


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
@pytest.mark.parametrize("length", [100])
def test_get_shape_from_observation_sequence(
    observation_shape: Shape, length: int
) -> None:
    observations = create_observations(observation_shape, length)
    assert (
        tuple(get_shape_from_observation_sequence(observations))
        == observation_shape
    )


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_get_dtype_from_observation(
    observation_shape: Shape, dtype: DType
) -> None:
    observation = create_observation(observation_shape, dtype=dtype)
    dtypes = get_dtype_from_observation(observation)
    if isinstance(dtypes, np.dtype):
        assert dtypes == dtype
    elif isinstance(dtypes, (list, tuple)):
        for t in dtypes:
            assert t == dtype


@pytest.mark.parametrize("observation_shape", [(4,), ((4,), (8,)), (3, 84, 84)])
@pytest.mark.parametrize("length", [100])
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_get_dtype_from_observation_sequence(
    observation_shape: Shape, length: int, dtype: DType
) -> None:
    observations = create_observations(observation_shape, length, dtype=dtype)
    dtypes = get_dtype_from_observation_sequence(observations)
    if isinstance(dtypes, np.dtype):
        assert dtypes == dtype
    elif isinstance(dtypes, (list, tuple)):
        for t in dtypes:
            assert t == dtype


@pytest.mark.parametrize("shape", [(4,), ((4,), (8,)), (3, 84, 84)])
def test_check_dtype(shape: Shape) -> None:
    array = create_observation(shape)
    if isinstance(shape[0], tuple):
        dtype = array[0].dtype
    else:
        assert isinstance(array, np.ndarray)
        dtype = array.dtype
    assert check_dtype(array, dtype)
    assert not check_dtype(array, np.uint8)


def test_check_non_1d_array() -> None:
    array1 = create_observation((4,))
    assert not check_non_1d_array(array1)

    array2 = create_observation((32, 4))
    assert check_non_1d_array(array2)

    array3 = create_observation(((32, 4), (4,)))
    assert not check_non_1d_array(array3)

    array4 = create_observation(((32, 4), (32, 4)))
    assert check_non_1d_array(array4)


@pytest.mark.parametrize("shape", [(4,), ((4,), (8,))])
def test_cast_recursively(shape: Shape) -> None:
    observation = create_observation(shape)

    casted_observation = cast_recursively(observation, np.uint8)

    assert check_dtype(casted_observation, np.uint8)


def test_detect_action_space() -> None:
    continuous_actions = np.random.random((100, 2))
    assert detect_action_space(continuous_actions) == ActionSpace.CONTINUOUS

    discrete_actions = np.random.randint(4, size=(100, 1))
    assert detect_action_space(discrete_actions) == ActionSpace.DISCRETE


def test_detect_action_space_from_env() -> None:
    env: Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]] = gym.make(
        "CartPole-v1"
    )
    assert detect_action_space_from_env(env) == ActionSpace.DISCRETE

    env = gym.make("Pendulum-v1")
    assert detect_action_space_from_env(env) == ActionSpace.CONTINUOUS

    env = gymnasium.make("CartPole-v1")
    assert detect_action_space_from_env(env) == ActionSpace.DISCRETE

    env = gymnasium.make("Pendulum-v1")
    assert detect_action_space_from_env(env) == ActionSpace.CONTINUOUS


def test_detect_action_size_from_env() -> None:
    env: Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]] = gym.make(
        "CartPole-v1"
    )
    assert detect_action_size_from_env(env) == 2

    env = gym.make("Pendulum-v1")
    assert detect_action_size_from_env(env) == 1

    env = gymnasium.make("CartPole-v1")
    assert detect_action_size_from_env(env) == 2

    env = gymnasium.make("Pendulum-v1")
    assert detect_action_size_from_env(env) == 1
