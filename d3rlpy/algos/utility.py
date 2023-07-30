from typing import Any

from gym.spaces import Box, Discrete
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Discrete as GymnasiumDiscrete

from ..base import LearnableBase
from ..constants import (
    CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR,
    DISCRETE_ACTION_SPACE_MISMATCH_ERROR,
    ActionSpace,
)
from ..dataset import DatasetInfo, ReplayBuffer
from ..envs import GymEnv
from ..logging import LOG

__all__ = ["assert_action_space_with_dataset", "assert_action_space_with_env"]


def assert_action_space_with_dataset(
    algo: LearnableBase[Any, Any], dataset_info: DatasetInfo
) -> None:
    if algo.get_action_type() == ActionSpace.BOTH:
        pass
    elif dataset_info.action_space == ActionSpace.DISCRETE:
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR


def assert_action_space_with_env(
    algo: LearnableBase[Any, Any], env: GymEnv
) -> None:
    if isinstance(env.action_space, (Box, GymnasiumBox)):
        assert (
            algo.get_action_type() == ActionSpace.CONTINUOUS
        ), CONTINUOUS_ACTION_SPACE_MISMATCH_ERROR
    elif isinstance(env.action_space, (Discrete, GymnasiumDiscrete)):
        assert (
            algo.get_action_type() == ActionSpace.DISCRETE
        ), DISCRETE_ACTION_SPACE_MISMATCH_ERROR
    else:
        action_space = type(env.action_space)
        raise ValueError(f"The action-space is not supported: {action_space}")


def build_scalers_with_transition_picker(
    algo: LearnableBase[Any, Any], dataset: ReplayBuffer
) -> None:
    # initialize observation scaler
    if algo.observation_scaler and not algo.observation_scaler.built:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )

    # initialize action scaler
    if algo.action_scaler and not algo.action_scaler.built:
        LOG.debug(
            "Fitting action scaler...",
            action_scaler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )

    # initialize reward scaler
    if algo.reward_scaler and not algo.reward_scaler.built:
        LOG.debug(
            "Fitting reward scaler...",
            reward_scaler=algo.reward_scaler.get_type(),
        )
        algo.reward_scaler.fit_with_transition_picker(
            dataset.episodes, dataset.transition_picker
        )


def build_scalers_with_trajectory_slicer(
    algo: LearnableBase[Any, Any], dataset: ReplayBuffer
) -> None:
    # initialize observation scaler
    if algo.observation_scaler and not algo.observation_scaler.built:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_trajectory_slicer(
            dataset.episodes, dataset.trajectory_slicer
        )

    # initialize action scaler
    if algo.action_scaler and not algo.action_scaler.built:
        LOG.debug(
            "Fitting action scaler...",
            action_scaler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_trajectory_slicer(
            dataset.episodes, dataset.trajectory_slicer
        )

    # initialize reward scaler
    if algo.reward_scaler and not algo.reward_scaler.built:
        LOG.debug(
            "Fitting reward scaler...",
            reward_scaler=algo.reward_scaler.get_type(),
        )
        algo.reward_scaler.fit_with_trajectory_slicer(
            dataset.episodes, dataset.trajectory_slicer
        )


def build_scalers_with_env(
    algo: LearnableBase[Any, Any],
    env: GymEnv,
) -> None:
    # initialize observation scaler
    if algo.observation_scaler and not algo.observation_scaler.built:
        LOG.debug(
            "Fitting observation scaler...",
            observation_scaler=algo.observation_scaler.get_type(),
        )
        algo.observation_scaler.fit_with_env(env)

    # initialize action scaler
    if algo.action_scaler and not algo.action_scaler.built:
        LOG.debug(
            "Fitting action scaler...",
            action_scler=algo.action_scaler.get_type(),
        )
        algo.action_scaler.fit_with_env(env)
