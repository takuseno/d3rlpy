from typing import Any, Sequence, TypeVar, Union, cast

import numpy as np

from ..constants import ActionSpace
from .types import Observation, ObservationSequence, Shape

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
    "check_dtype",
    "check_non_1d_array",
    "cast_recursively",
    "detect_action_space",
    "is_tuple_shape",
    "cast_flat_shape",
    "cast_tuple_shape",
    "get_axis_size",
    "get_batch_dim",
]


def retrieve_observation(
    observations: ObservationSequence, index: int
) -> Observation:
    if isinstance(observations, np.ndarray):
        return observations[index]
    elif isinstance(observations, (list, tuple)):
        return [obs[index] for obs in observations]
    else:
        raise ValueError(f"invalid observations type: {type(observations)}")


def create_zero_observation(observation: Observation) -> Observation:
    if isinstance(observation, np.ndarray):
        return np.zeros_like(observation)
    elif isinstance(observation, (list, tuple)):
        return [np.zeros_like(observation[i]) for i in range(len(observation))]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


def slice_observations(
    observations: ObservationSequence, start: int, end: int
) -> ObservationSequence:
    if isinstance(observations, np.ndarray):
        return observations[start:end]
    elif isinstance(observations, (list, tuple)):
        return [obs[start:end] for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


def batch_pad_array(array: np.ndarray, pad_size: int) -> np.ndarray:
    batch_size = array.shape[0]
    shape = array.shape[1:]
    padded_array = np.zeros((pad_size + batch_size, *shape), dtype=array.dtype)
    padded_array[-batch_size:] = array
    return padded_array


def batch_pad_observations(
    observations: ObservationSequence, pad_size: int
) -> ObservationSequence:
    if isinstance(observations, np.ndarray):
        return batch_pad_array(observations, pad_size)
    elif isinstance(observations, (list, tuple)):
        padded_observations = [
            batch_pad_observations(obs, pad_size) for obs in observations
        ]
        return cast(ObservationSequence, padded_observations)
    else:
        raise ValueError(f"invalid observations type: {type(observations)}")


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

    def squeeze_batch_dim(array: np.ndarray) -> np.ndarray:
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


def get_shape_from_observation(observation: Observation) -> Shape:
    if isinstance(observation, np.ndarray):
        return observation.shape  # type: ignore
    elif isinstance(observation, (list, tuple)):
        return [obs.shape for obs in observation]
    else:
        raise ValueError(f"invalid observation type: {type(observation)}")


def get_shape_from_observation_sequence(
    observations: ObservationSequence,
) -> Shape:
    if isinstance(observations, np.ndarray):
        return observations.shape[1:]  # type: ignore
    elif isinstance(observations, (list, tuple)):
        return [obs.shape[1:] for obs in observations]
    else:
        raise ValueError(f"invalid observation type: {type(observations)}")


def check_dtype(
    array: Union[np.ndarray, Sequence[np.ndarray]], dtype: Any
) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.dtype == dtype for v in array)
    elif isinstance(array, np.ndarray):
        return array.dtype == dtype  # type: ignore
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def check_non_1d_array(array: Union[np.ndarray, Sequence[np.ndarray]]) -> bool:
    if isinstance(array, (list, tuple)):
        return all(v.ndim > 1 for v in array)
    elif isinstance(array, np.ndarray):
        return array.ndim > 1  # type: ignore
    else:
        raise ValueError(f"invalid array type: {type(array)}")


_T = TypeVar("_T")


def cast_recursively(array: _T, dtype: Any) -> _T:
    if isinstance(array, (list, tuple)):
        return [array[i].astype(dtype) for i in range(len(array))]  # type: ignore
    elif isinstance(array, np.ndarray):
        return array.astype(dtype)  # type: ignore
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def detect_action_space(actions: np.ndarray) -> ActionSpace:
    if np.all(np.array(actions, dtype=np.int32) == actions):
        return ActionSpace.DISCRETE
    else:
        return ActionSpace.CONTINUOUS


def is_tuple_shape(shape: Shape) -> bool:
    return isinstance(shape[0], (list, tuple))


def cast_tuple_shape(shape: Shape) -> Sequence[Sequence[int]]:
    assert is_tuple_shape(shape)
    return shape  # type: ignore


def cast_flat_shape(shape: Shape) -> Sequence[int]:
    assert not is_tuple_shape(shape)
    return shape  # type: ignore


def get_axis_size(
    array: Union[np.ndarray, Sequence[np.ndarray]], axis: int
) -> int:
    if isinstance(array, np.ndarray):
        return int(array.shape[axis])
    elif isinstance(array, (list, tuple)):
        sizes = list(map(lambda v: v.shape[axis], array))
        size = sizes[axis]
        assert np.all(np.array(sizes) == size)
        return int(size)
    else:
        raise ValueError(f"invalid array type: {type(array)}")


def get_batch_dim(array: Union[np.ndarray, Sequence[np.ndarray]]) -> int:
    return get_axis_size(array, axis=0)
