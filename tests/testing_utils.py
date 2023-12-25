from typing import Optional, Sequence, Tuple, cast, overload

import numpy as np
import torch

from d3rlpy.dataset import (
    Episode,
    PartialTrajectory,
    Transition,
    is_tuple_shape,
)
from d3rlpy.preprocessing import (
    ActionScaler,
    MinMaxActionScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
    ObservationScaler,
    RewardScaler,
    TupleObservationScaler,
)
from d3rlpy.torch_utility import convert_to_torch_recursively
from d3rlpy.types import (
    DType,
    Float32NDArray,
    NDArray,
    Observation,
    ObservationSequence,
    Shape,
    TorchObservation,
)


@overload
def create_observation(
    observation_shape: Sequence[int], dtype: DType = np.float32
) -> NDArray:
    ...


@overload
def create_observation(
    observation_shape: Sequence[Sequence[int]], dtype: DType = np.float32
) -> Sequence[NDArray]:
    ...


def create_observation(
    observation_shape: Shape, dtype: DType = np.float32
) -> Observation:
    observation: Observation
    if isinstance(observation_shape[0], (list, tuple)):
        observation = [
            np.random.random(shape).astype(dtype) for shape in observation_shape
        ]
    else:
        observation = np.random.random(observation_shape).astype(dtype)
    return observation


@overload
def create_torch_observation(
    observation_shape: Sequence[int], dtype: DType = np.float32
) -> torch.Tensor:
    ...


@overload
def create_torch_observation(
    observation_shape: Sequence[Sequence[int]], dtype: DType = np.float32
) -> Sequence[torch.Tensor]:
    ...


def create_torch_observation(
    observation_shape: Shape, dtype: DType = np.float32
) -> TorchObservation:
    return convert_to_torch_recursively(
        create_observation(observation_shape, dtype), "cpu"
    )


@overload
def create_observations(
    observation_shape: Sequence[int], length: int, dtype: DType = np.float32
) -> NDArray:
    ...


@overload
def create_observations(
    observation_shape: Sequence[Sequence[int]],
    length: int,
    dtype: DType = np.float32,
) -> Sequence[NDArray]:
    ...


def create_observations(
    observation_shape: Shape, length: int, dtype: DType = np.float32
) -> ObservationSequence:
    observations: ObservationSequence
    if isinstance(observation_shape[0], (list, tuple)):
        observations = [
            np.random.random((length, *shape)).astype(dtype)
            for shape in cast(Sequence[Sequence[int]], observation_shape)
        ]
    else:
        observations = np.random.random((length, *observation_shape)).astype(
            dtype
        )
    return observations


@overload
def create_torch_observations(
    observation_shape: Sequence[int], length: int, dtype: DType = np.float32
) -> torch.Tensor:
    ...


@overload
def create_torch_observations(
    observation_shape: Sequence[Sequence[int]],
    length: int,
    dtype: DType = np.float32,
) -> Sequence[torch.Tensor]:
    ...


def create_torch_observations(
    observation_shape: Shape, length: int, dtype: DType = np.float32
) -> TorchObservation:
    return convert_to_torch_recursively(
        create_observations(observation_shape, length, dtype), "cpu"
    )


@overload
def create_torch_batched_observations(
    observation_shape: Sequence[int],
    batch_size: int,
    length: int,
    dtype: DType = np.float32,
) -> torch.Tensor:
    ...


@overload
def create_torch_batched_observations(
    observation_shape: Sequence[Sequence[int]],
    batch_size: int,
    length: int,
    dtype: DType = np.float32,
) -> Sequence[torch.Tensor]:
    ...


def create_torch_batched_observations(
    observation_shape: Shape,
    batch_size: int,
    length: int,
    dtype: DType = np.float32,
) -> TorchObservation:
    observations = convert_to_torch_recursively(
        create_observations(observation_shape, batch_size * length, dtype),
        "cpu",
    )
    if isinstance(observations, torch.Tensor):
        return observations.view(batch_size, length, *observation_shape)
    else:
        return [
            o.view(batch_size, length, *s)
            for o, s in zip(observations, observation_shape)
        ]


def create_episode(
    observation_shape: Shape,
    action_size: int,
    length: int,
    discrete_action: bool = False,
    terminated: bool = False,
) -> Episode:
    observations = create_observations(observation_shape, length)

    actions: NDArray
    if discrete_action:
        actions = np.random.randint(action_size, size=(length, 1))
    else:
        actions = np.random.random((length, action_size))

    return Episode(
        observations=observations,
        actions=actions,
        rewards=np.random.random((length, 1)).astype(np.float32),
        terminated=terminated,
    )


def create_transition(
    observation_shape: Shape,
    action_size: int,
    discrete_action: bool = False,
    terminated: bool = False,
) -> Transition:
    observation: Observation
    next_observation: Observation
    if isinstance(observation_shape[0], (list, tuple)):
        observation = [
            np.random.random(shape).astype(np.float32)
            for shape in observation_shape
        ]
        next_observation = [
            np.random.random(shape).astype(np.float32)
            for shape in observation_shape
        ]
    else:
        observation = np.random.random(observation_shape).astype(np.float32)
        next_observation = np.random.random(observation_shape).astype(
            np.float32
        )

    action: NDArray
    if discrete_action:
        action = np.random.randint(action_size, size=(1,))
    else:
        action = np.random.random(action_size).astype(np.float32)

    return Transition(
        observation=observation,
        action=action,
        reward=np.random.random(1).astype(np.float32),
        next_observation=next_observation,
        return_to_go=np.random.random(1).astype(np.float32),
        terminal=1.0 if terminated else 0.0,
        interval=1,
    )


def create_partial_trajectory(
    observation_shape: Shape,
    action_size: int,
    length: int,
    discrete_action: bool = False,
) -> PartialTrajectory:
    observations = create_observations(observation_shape, length)

    actions: NDArray
    if discrete_action:
        actions = np.random.randint(action_size, size=(length, 1))
    else:
        actions = np.random.random((length, action_size))

    rewards: Float32NDArray = np.random.random((length, 1)).astype(np.float32)

    return PartialTrajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        returns_to_go=np.reshape(np.cumsum(rewards), [-1, 1]),
        terminals=np.zeros((length, 1), dtype=np.float32),
        timesteps=np.arange(length),
        masks=np.ones(length, dtype=np.float32),
        length=length,
    )


def create_scaler_tuple(
    name: Optional[str],
    observation_shape: Shape,
) -> Tuple[
    Optional[ObservationScaler], Optional[ActionScaler], Optional[RewardScaler]
]:
    if name is None:
        return None, None, None
    observation_scaler: ObservationScaler
    if is_tuple_shape(observation_shape):
        observation_scaler = TupleObservationScaler(
            [MinMaxObservationScaler() for _ in range(len(observation_shape))]
        )
    else:
        observation_scaler = MinMaxObservationScaler()
    return observation_scaler, MinMaxActionScaler(), MinMaxRewardScaler()
