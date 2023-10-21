from typing import Sequence, TypeVar, Union, overload

import numpy as np
from gym.spaces import Box, Discrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete

from ..constants import ActionSpace
from ..envs.types import GymEnv
from ..types import DType, NDArray, Observation, ObservationSequence, Shape

__all__ = [
    "retrieve_observation",
    "create_zero_observation",
    "slice_observations",
    "batch_pad_array",
    "batch_pad_observations",
    "stack_recent_observations",
    "stack_observations",
    "get_shape_from_observation",
    "get_shape_from_observation_sequence",
    "get_dtype_from_observation",
    "get_dtype_from_observation_sequence",
    "check_dtype",
    "check_non_1d_array",
    "cast_recursively",
    "detect_action_space",
    "detect_action_space_from_env",
    "detect_action_size_from_env",
    "is_tuple_shape",
    "cast_flat_shape",
    "cast_tuple_shape",
    "get_axis_size",
    "get_batch_dim",
]


@overload
def retrieve_observation(observations: NDArray, index: int) -> NDArray:
    ...


@overload
def retrieve_observation(
    observations: Sequence[NDArray], index: int
) -> Sequence[NDArray]:
    ...


def retrieve_observation(
    observations: ObservationSequence, index: int
) -> Observation:
    if isinstance(observations, np.ndarray):
        return observations[index]  # type: ignore
    elif isinstance(observations, (list, tuple)):
        return [obs[index] for obs in observations]
    else:
        raise ValueError(f"invalid observations type: {type(observations)}")


@overload
def create_zero_observation(observation: NDArray) -> NDArray:
    ...


@overload
def create_zero_observation(
    observation: Sequence[NDArray],
) -> Sequence[NDArray]:
    ...


def create_zero_observation(observation: Observation) -> Observation:
    if isinstance(observation, np.ndarray):
        return np.zeros_like(observation)
    elif isinstance(observation, (list, tuple)):
        return [np.zeros_like(observation[i]) for i in range(len(observation))]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


@overload
def slice_observations(observations: NDArray, start: int, end: int) -> NDArray:
    ...


@overload
def slice_observations(
    observations: Sequence[NDArray], start: int, end: int
) -> Sequence[NDArray]:
    ...


def slice_observations(
    observations: ObservationSequence, start: int, end: int
) -> ObservationSequence:
    if isinstance(observations, np.ndarray):
        return observations[start:end]  # type: ignore
    elif isinstance(observations, (list, tuple)):
        return [obs[start:end] for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


def batch_pad_array(array: NDArray, pad_size: int) -> NDArray:
    batch_size = array.shape[0]
    shape = array.shape[1:]
    padded_array = np.zeros((pad_size + batch_size, *shape), dtype=array.dtype)
    padded_array[-batch_size:] = array
    return padded_array


@overload
def batch_pad_observations(observations: NDArray, pad_size: int) -> NDArray:
    ...


@overload
def batch_pad_observations(
    observations: Sequence[NDArray], pad_size: int
) -> Sequence[NDArray]:
    ...


def batch_pad_observations(
    observations: ObservationSequence, pad_size: int
) -> ObservationSequence:
    if isinstance(observations, np.ndarray):
        return batch_pad_array(observations, pad_size)
    elif isinstance(observations, (list, tuple)):
        padded_observations = [
            batch_pad_observations(obs, pad_size) for obs in observations
        ]
        return padded_observations
    else:
        raise ValueError(f"invalid observations type: {type(observations)}")


@overload
def stack_recent_observations(
    observations: NDArray, index: int, n_frames: int
) -> NDArray:
    ...


@overload
def stack_recent_observations(
    observations: Sequence[NDArray], index: int, n_frames: int
) -> Sequence[NDArray]:
    ...


def stack_recent_observations(
    observations: ObservationSequence, index: int, n_frames: int
) -> Observation:
    start = max(index - n_frames + 1, 0)
    end = index + 1
    # (B, C, ...)
    observation_seq = slice_observations(observations, start, end)
    if end - start < n_frames:
        observation_seq = batch_pad_observations(
            observation_seq, n_frames - (end - start)
        )

    def squeeze_batch_dim(array: NDArray) -> NDArray:
        shape = array.shape
        batch_size = shape[0]
        channel_size = shape[1]
        rest_shape = shape[2:]
        return np.reshape(array, [batch_size * channel_size, *rest_shape])

    # (B, C, ...) -> (B * C, ...)
    if isinstance(observation_seq, np.ndarray):
        return squeeze_batch_dim(observation_seq)
    elif isinstance(observation_seq, (list, tuple)):
        return [squeeze_batch_dim(obs) for obs in observation_seq]
    else:
        raise ValueError(f"invalid observation type: {type(observation_seq)}")


@overload
def stack_observations(observations: Sequence[NDArray]) -> NDArray:
    ...


@overload
def stack_observations(
    observations: Sequence[Sequence[NDArray]],
) -> Sequence[NDArray]:
    ...


@overload
def stack_observations(observations: Sequence[Observation]) -> Observation:
    ...


def stack_observations(observations: Sequence[Observation]) -> Observation:
    if isinstance(observations[0], (list, tuple)):
        obs_kinds = len(observations[0])
        return [
            np.stack([obs[i] for obs in observations], axis=0)
            for i in range(obs_kinds)
        ]
    elif isinstance(observations[0], np.ndarray):
        return np.stack(observations, axis=0)
    else:
        raise ValueError(f"invalid observation type: {type(observations[0])}")


@overload
def get_shape_from_observation(observation: NDArray) -> Sequence[int]:
    ...


@overload
def get_shape_from_observation(
    observation: Sequence[NDArray],
) -> Sequence[Sequence[int]]:
    ...


def get_shape_from_observation(observation: Observation) -> Shape:
    if isinstance(observation, np.ndarray):
        return observation.shape
    elif isinstance(observation, (list, tuple)):
        return [obs.shape for obs in observation]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


@overload
def get_shape_from_observation_sequence(
    observations: NDArray,
) -> Sequence[int]:
    ...


@overload
def get_shape_from_observation_sequence(
    observations: Sequence[NDArray],
) -> Sequence[Sequence[int]]:
    ...


def get_shape_from_observation_sequence(
    observations: ObservationSequence,
) -> Shape:
    if isinstance(observations, np.ndarray):
        return observations.shape[1:]
    elif isinstance(observations, (list, tuple)):
        return [obs.shape[1:] for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


@overload
def get_dtype_from_observation(observation: NDArray) -> DType:
    ...


@overload
def get_dtype_from_observation(
    observation: Sequence[NDArray],
) -> Sequence[DType]:
    ...


def get_dtype_from_observation(
    observation: Observation,
) -> Union[DType, Sequence[DType]]:
    if isinstance(observation, np.ndarray):
        return observation.dtype
    elif isinstance(observation, (list, tuple)):
        return [obs.dtype for obs in observation]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


@overload
def get_dtype_from_observation_sequence(
    observations: NDArray,
) -> DType:
    ...


@overload
def get_dtype_from_observation_sequence(
    observations: Sequence[NDArray],
) -> Sequence[DType]:
    ...


def get_dtype_from_observation_sequence(
    observations: ObservationSequence,
) -> Union[DType, Sequence[DType]]:
    if isinstance(observations, np.ndarray):
        return observations.dtype
    elif isinstance(observations, (list, tuple)):
        return [obs.dtype for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


def check_dtype(array: Union[NDArray, Sequence[NDArray]], dtype: DType) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.dtype == dtype for v in array)
    elif isinstance(array, np.ndarray):
        return array.dtype == dtype
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def check_non_1d_array(array: Union[NDArray, Sequence[NDArray]]) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.ndim > 1 for v in array)
    elif isinstance(array, np.ndarray):
        return array.ndim > 1
    else:
        raise ValueError(f"invalid array type: {type(array)}")


_T = TypeVar("_T")


def cast_recursively(array: _T, dtype: DType) -> _T:
    if isinstance(array, (list, tuple)):
        return [array[i].astype(dtype) for i in range(len(array))]  # type: ignore
    elif isinstance(array, np.ndarray):
        return array.astype(dtype)  # type: ignore
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def detect_action_space(actions: NDArray) -> ActionSpace:
    if np.all(np.array(actions, dtype=np.int32) == actions):
        return ActionSpace.DISCRETE
    else:
        return ActionSpace.CONTINUOUS


def detect_action_space_from_env(env: GymEnv) -> ActionSpace:
    if isinstance(env.action_space, (Box, GymnasiumBox)):
        action_space = ActionSpace.CONTINUOUS
    elif isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        action_space = ActionSpace.DISCRETE
    else:
        raise ValueError(f"Unsupported action_space: {type(env.action_space)}")
    return action_space


def detect_action_size_from_env(env: GymEnv) -> int:
    if isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        action_size = env.action_space.n
    elif isinstance(env.action_space, (Box, GymnasiumBox)):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError(f"Unsupported action_space: {type(env.action_space)}")
    return int(action_size)


def is_tuple_shape(shape: Shape) -> bool:
    return isinstance(shape[0], (list, tuple))


def cast_tuple_shape(shape: Shape) -> Sequence[Sequence[int]]:
    assert is_tuple_shape(shape)
    return shape  # type: ignore


def cast_flat_shape(shape: Shape) -> Sequence[int]:
    assert not is_tuple_shape(shape)
    return shape  # type: ignore


def get_axis_size(array: Union[NDArray, Sequence[NDArray]], axis: int) -> int:
    if isinstance(array, np.ndarray):
        return int(array.shape[axis])
    elif isinstance(array, (list, tuple)):
        sizes = list(map(lambda v: v.shape[axis], array))
        size = sizes[axis]
        assert np.all(np.array(sizes) == size)
        return int(size)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def get_batch_dim(array: Union[NDArray, Sequence[NDArray]]) -> int:
    return get_axis_size(array, axis=0)
