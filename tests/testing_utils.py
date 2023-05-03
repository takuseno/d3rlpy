import numpy as np

from d3rlpy.dataset import Episode, PartialTrajectory, Transition
from d3rlpy.preprocessing import (
    MinMaxActionScaler,
    MinMaxObservationScaler,
    MinMaxRewardScaler,
)


def create_observation(observation_shape, dtype=np.float32):
    if isinstance(observation_shape[0], (list, tuple)):
        observation = [
            np.random.random(shape).astype(dtype) for shape in observation_shape
        ]
    else:
        observation = np.random.random(observation_shape).astype(dtype)
    return observation


def create_observations(observation_shape, length, dtype=np.float32):
    if isinstance(observation_shape[0], (list, tuple)):
        observations = [
            np.random.random((length,) + shape).astype(dtype)
            for shape in observation_shape
        ]
    else:
        observations = np.random.random((length,) + observation_shape).astype(
            dtype
        )
    return observations


def create_episode(
    observation_shape,
    action_size,
    length,
    discrete_action=False,
    terminated=False,
):
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
    observation_shape, action_size, discrete_action=False, terminated=False
):
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
    observation_shape, action_size, length, discrete_action=False
):
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


def create_scaler_tuple(name):
    if name is None:
        return None, None, None
    return MinMaxObservationScaler(), MinMaxActionScaler(), MinMaxRewardScaler()
