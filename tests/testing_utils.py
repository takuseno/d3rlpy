from typing import Optional, Sequence, Tuple, cast

import numpy as np

from d3rlpy.dataset import (
    Episode,
    Observation,
    ObservationSequence,
    PartialTrajectory,
    Shape,
    Transition,
)
from d3rlpy.preprocessing import (
    ActionScaler,
    MinMaxActionScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
    ObservationScaler,
    RewardScaler,
)


def create_observation(
    observation_shape: Shape, dtype: np.dtype = np.float32
) -> Observation:
    if isinstance(observation_shape[0], (list, tuple)):
        observation = [
            np.random.random(shape).astype(dtype) for shape in observation_shape
        ]
    else:
        observation = np.random.random(observation_shape).astype(dtype)
    return observation


def create_observations(
    observation_shape: Shape, length: int, dtype: np.dtype = np.float32
) -> ObservationSequence:
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


def create_episode(
    observation_shape: Shape,
    action_size: int,
    length: int,
    discrete_action: bool = False,
    terminated: bool = False,
) -> Episode:
    observations = create_observations(observation_shape, length)

    if discrete_action:
        actions = np.random.randint(action_size, size=(length, 1))
    else:
        actions = np.random.random((length, action_size))

    return Episode(
        observations=observations,
        actions=actions,
        rewards=np.random.random((length, 1)),
        terminated=terminated,
    )


def create_transition(
    observation_shape: Shape,
    action_size: int,
    discrete_action: bool = False,
    terminated: bool = False,
) -> Transition:
    if isinstance(observation_shape[0], (list, tuple)):
        observation = [np.random.random(shape) for shape in observation_shape]
        next_observation = [
            np.random.random(shape) for shape in observation_shape
        ]
    else:
        observation = np.random.random(observation_shape)
        next_observation = np.random.random(observation_shape)

    if discrete_action:
        action = np.random.randint(action_size, size=(1,))
    else:
        action = np.random.random(action_size)

    return Transition(
        observation=observation,
        action=action,
        reward=np.random.random(1),
        next_observation=next_observation,
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

    if discrete_action:
        actions = np.random.randint(action_size, size=(length, 1))
    else:
        actions = np.random.random((length, action_size))

    rewards = np.random.random((length, 1))

    return PartialTrajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        returns_to_go=np.reshape(np.cumsum(rewards), [-1, 1]),
        terminals=np.zeros((length, 1)),
        timesteps=np.arange(length),
        masks=np.ones(length),
        length=length,
    )


def create_scaler_tuple(
    name: str,
) -> Tuple[
    Optional[ObservationScaler], Optional[ActionScaler], Optional[RewardScaler]
]:
    if name is None:
        return None, None, None
    return MinMaxObservationScaler(), MinMaxActionScaler(), MinMaxRewardScaler()
