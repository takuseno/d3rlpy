import enum
import os
import random
import re
from typing import Any, Optional
from urllib import request

import gym
import gymnasium
import numpy as np
from gym.wrappers.time_limit import TimeLimit
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Dict as GymnasiumDictSpace
from gymnasium.wrappers import TimeLimit as GymnasiumTimeLimit

from .dataset import (
    BasicTrajectorySlicer,
    BasicTransitionPicker,
    Episode,
    EpisodeGenerator,
    FrameStackTrajectorySlicer,
    FrameStackTransitionPicker,
    InfiniteBuffer,
    MDPDataset,
    ReplayBuffer,
    TrajectorySlicerProtocol,
    TransitionPickerProtocol,
    create_infinite_replay_buffer,
    load_v1,
)
from .envs import ChannelFirst, FrameStack, GoalConcatWrapper
from .logging import LOG
from .types import NDArray, UInt8NDArray

__all__ = [
    "DATA_DIRECTORY",
    "DROPBOX_URL",
    "CARTPOLE_URL",
    "CARTPOLE_RANDOM_URL",
    "PENDULUM_URL",
    "PENDULUM_RANDOM_URL",
    "get_cartpole",
    "get_pendulum",
    "get_atari",
    "get_atari_transitions",
    "get_d4rl",
    "get_dataset",
]

DATA_DIRECTORY = "d3rlpy_data"
DROPBOX_URL = "https://www.dropbox.com/s"
CARTPOLE_URL = f"{DROPBOX_URL}/uep0lzlhxpi79pd/cartpole_v1.1.0.h5?dl=1"
CARTPOLE_RANDOM_URL = f"{DROPBOX_URL}/4lgai7tgj84cbov/cartpole_random_v1.1.0.h5?dl=1"  # noqa: E501
PENDULUM_URL = f"{DROPBOX_URL}/ukkucouzys0jkfs/pendulum_v1.1.0.h5?dl=1"
PENDULUM_RANDOM_URL = f"{DROPBOX_URL}/hhbq9i6ako24kzz/pendulum_random_v1.1.0.h5?dl=1"  # noqa: E501


def get_cartpole(
    dataset_type: str = "replay",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
) -> tuple[ReplayBuffer, gym.Env[NDArray, int]]:
    """Returns cartpole dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/cartpole.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if dataset_type == "replay":
        url = CARTPOLE_URL
        file_name = "cartpole_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = CARTPOLE_RANDOM_URL
        file_name = "cartpole_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    # download dataset
    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Downloading cartpole.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )

    # environment
    env = gym.make("CartPole-v1", render_mode=render_mode)

    return dataset, env


def get_pendulum(
    dataset_type: str = "replay",
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
) -> tuple[ReplayBuffer, gym.Env[NDArray, NDArray]]:
    """Returns pendulum dataset and environment.

    The dataset is automatically downloaded to ``d3rlpy_data/pendulum.h5`` if
    it does not exist.

    Args:
        dataset_type: dataset type. Available options are
            ``['replay', 'random']``.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if dataset_type == "replay":
        url = PENDULUM_URL
        file_name = "pendulum_replay_v1.1.0.h5"
    elif dataset_type == "random":
        url = PENDULUM_RANDOM_URL
        file_name = "pendulum_random_v1.1.0.h5"
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}.")

    data_path = os.path.join(DATA_DIRECTORY, file_name)

    if not os.path.exists(data_path):
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        print(f"Donwloading pendulum.pkl into {data_path}...")
        request.urlretrieve(url, data_path)

    # load dataset
    with open(data_path, "rb") as f:
        episodes = load_v1(f)
    dataset = ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        transition_picker=transition_picker,
        trajectory_slicer=trajectory_slicer,
    )

    # environment
    env = gym.make("Pendulum-v1", render_mode=render_mode)

    return dataset, env


def _stack_frames(episode: Episode, num_stack: int) -> Episode:
    assert isinstance(episode.observations, np.ndarray)
    episode_length = episode.observations.shape[0]
    observations: UInt8NDArray = np.zeros(
        (episode_length, num_stack, 84, 84),
        dtype=np.uint8,
    )
    for i in range(num_stack):
        pad_size = num_stack - i - 1
        if pad_size > 0:
            observations[pad_size:, i] = np.reshape(
                episode.observations[:-pad_size], [-1, 84, 84]
            )
        else:
            observations[:, i] = np.reshape(episode.observations, [-1, 84, 84])
    return Episode(
        observations=observations,
        actions=episode.actions.copy(),
        rewards=episode.rewards.copy(),
        terminated=episode.terminated,
    )


def get_atari(
    env_name: str,
    num_stack: Optional[int] = None,
    sticky_action: bool = True,
    pre_stack: bool = False,
    render_mode: Optional[str] = None,
) -> tuple[ReplayBuffer, gym.Env[NDArray, int]]:
    """Returns atari dataset and envrironment.

    The dataset is provided through d4rl-atari. See more details including
    available dataset from its GitHub page.

    .. code-block:: python

        from d3rlpy.datasets import get_atari

        dataset, env = get_atari('breakout-mixed-v0')

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        env_name: environment id of d4rl-atari dataset.
        num_stack: the number of frames to stack (only applied to env).
        sticky_action: Flag to enable sticky action.
        pre_stack: Flag to pre-stack observations. If this is ``False``,
            ``FrameStackTransitionPicker`` and ``FrameStackTrajectorySlicer``
            will be used to stack observations at sampling-time.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import d4rl_atari  # type: ignore # noqa

        env = gym.make(
            env_name,
            render_mode=render_mode,
            sticky_action=sticky_action,
        )
        raw_dataset = env.get_dataset()  # type: ignore
        episode_generator = EpisodeGenerator(**raw_dataset)
        episodes = episode_generator()

        if pre_stack:
            stacked_episodes = []
            for episode in episodes:
                assert num_stack is not None
                stacked_episode = _stack_frames(episode, num_stack)
                stacked_episodes.append(stacked_episode)
            episodes = stacked_episodes

        picker: TransitionPickerProtocol
        slicer: TrajectorySlicerProtocol
        if num_stack is None or pre_stack:
            picker = BasicTransitionPicker()
            slicer = BasicTrajectorySlicer()
        else:
            picker = FrameStackTransitionPicker(num_stack or 1)
            slicer = FrameStackTrajectorySlicer(num_stack or 1)

        dataset = create_infinite_replay_buffer(
            episodes=episodes,
            transition_picker=picker,
            trajectory_slicer=slicer,
        )
        if num_stack:
            env = FrameStack(env, num_stack=num_stack)
        else:
            env = ChannelFirst(env)
        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n" "$ d3rlpy install d4rl_atari"
        ) from e


def get_atari_transitions(
    game_name: str,
    fraction: float = 0.01,
    index: int = 0,
    num_stack: Optional[int] = None,
    sticky_action: bool = True,
    pre_stack: bool = False,
    render_mode: Optional[str] = None,
) -> tuple[ReplayBuffer, gym.Env[NDArray, int]]:
    """Returns atari dataset as a list of Transition objects and envrironment.

    The dataset is provided through d4rl-atari.
    The difference from ``get_atari`` function is that this function will
    sample transitions from all epochs.
    This function is necessary for reproducing Atari experiments.

    .. code-block:: python

        from d3rlpy.datasets import get_atari_transitions

        # get 1% of transitions from all epochs (1M x 50 epoch x 1% = 0.5M)
        dataset, env = get_atari_transitions('breakout', fraction=0.01)

    References:
        * https://github.com/takuseno/d4rl-atari

    Args:
        game_name: Atari 2600 game name in lower_snake_case.
        fraction: fraction of sampled transitions.
        index: index to specify which trial to load.
        num_stack: the number of frames to stack (only applied to env).
        sticky_action: Flag to enable sticky action.
        pre_stack: Flag to pre-stack observations. If this is ``False``,
            ``FrameStackTransitionPicker`` and ``FrameStackTrajectorySlicer``
            will be used to stack observations at sampling-time.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of a list of :class:`d3rlpy.dataset.Transition` and gym
        environment.
    """
    try:
        import d4rl_atari  # noqa

        # each epoch consists of 1M steps
        num_transitions_per_epoch = int(1000000 * fraction)

        copied_episodes = []
        for i in range(50):
            env_name = f"{game_name}-epoch-{i + 1}-v{index}"
            LOG.info(f"Collecting {env_name}...")
            env = gym.make(
                env_name,
                sticky_action=sticky_action,
                render_mode=render_mode,
            )
            raw_dataset = env.get_dataset()  # type: ignore
            episode_generator = EpisodeGenerator(**raw_dataset)
            episodes = list(episode_generator())

            # copy episode data to release memory of unused data
            random.shuffle(episodes)
            num_data = 0
            for episode in episodes:
                if num_data >= num_transitions_per_epoch:
                    break

                assert isinstance(episode.observations, np.ndarray)
                copied_episode = Episode(
                    observations=episode.observations.copy(),
                    actions=episode.actions.copy(),
                    rewards=episode.rewards.copy(),
                    terminated=episode.terminated,
                )
                if pre_stack:
                    assert num_stack is not None
                    copied_episode = _stack_frames(copied_episode, num_stack)

                # trim episode
                if num_data + copied_episode.size() > num_transitions_per_epoch:
                    end = num_transitions_per_epoch - num_data
                    copied_episode = Episode(
                        observations=copied_episode.observations[:end],
                        actions=copied_episode.actions[:end],
                        rewards=copied_episode.rewards[:end],
                        terminated=False,
                    )

                copied_episodes.append(copied_episode)
                num_data += copied_episode.size()

        picker: TransitionPickerProtocol
        slicer: TrajectorySlicerProtocol
        if num_stack is None or pre_stack:
            picker = BasicTransitionPicker()
            slicer = BasicTrajectorySlicer()
        else:
            picker = FrameStackTransitionPicker(num_stack or 1)
            slicer = FrameStackTrajectorySlicer(num_stack or 1)

        dataset = ReplayBuffer(
            InfiniteBuffer(),
            episodes=copied_episodes,
            transition_picker=picker,
            trajectory_slicer=slicer,
        )

        if num_stack:
            env = FrameStack(env, num_stack=num_stack)
        else:
            env = ChannelFirst(env)

        return dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl-atari is not installed.\n" "$ d3rlpy install d4rl_atari"
        ) from e


def get_d4rl(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 1000,
) -> tuple[ReplayBuffer, gym.Env[NDArray, NDArray]]:
    """Returns d4rl dataset and envrironment.

    The dataset is provided through d4rl.

    .. code-block:: python

        from d3rlpy.datasets import get_d4rl

        dataset, env = get_d4rl('hopper-medium-v0')

    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl

    Args:
        env_name: environment id of d4rl dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).
        max_episode_steps: Maximum episode environmental steps.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import d4rl  # noqa
        from d4rl.pointmaze.maze_model import MazeEnv
        from d4rl.locomotion.wrappers import NormalizedBoxEnv
        from d4rl.utils.wrappers import (
            NormalizedBoxEnv as NormalizedBoxEnvFromUtils,
        )

        env = gym.make(env_name)
        raw_dataset: dict[str, NDArray] = env.get_dataset()  # type: ignore

        observations = raw_dataset["observations"]
        actions = raw_dataset["actions"]
        rewards = raw_dataset["rewards"]
        terminals = raw_dataset["terminals"]
        timeouts = raw_dataset["timeouts"]

        dataset = MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

        # remove incompatible wrappers
        wrapped_env = env.env.env.env  # type: ignore
        if isinstance(
            wrapped_env, (NormalizedBoxEnv, NormalizedBoxEnvFromUtils)
        ):
            unwrapped_env: gym.Env[Any, Any] = wrapped_env.wrapped_env
            unwrapped_env.render_mode = render_mode  # overwrite
        elif isinstance(wrapped_env, MazeEnv):
            wrapped_env.render_mode = render_mode  # overwrite
        else:
            wrapped_env.env.render_mode = render_mode  # overwrite

        return dataset, TimeLimit(
            wrapped_env, max_episode_steps=max_episode_steps
        )
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n" "$ d3rlpy install d4rl"
        ) from e


class _MinariEnvType(enum.Enum):
    BOX = 0
    GOAL_CONDITIONED = 1


def get_minari(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
    tuple_observation: bool = False,
) -> tuple[ReplayBuffer, gymnasium.Env[Any, Any]]:
    """Returns minari dataset and envrironment.

    The dataset is provided through minari.

    .. code-block:: python
        from d3rlpy.datasets import get_minari
        dataset, env = get_minari('door-cloned-v1')

    Args:
        env_name: environment id of minari dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).
        tuple_observation: Flag to include goals as tuple element.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import minari

        _dataset = minari.load_dataset(env_name, download=True)
        env = _dataset.recover_environment()
        unwrapped_env = env.unwrapped
        unwrapped_env.render_mode = render_mode

        if isinstance(env.observation_space, GymnasiumBox):
            env_type = _MinariEnvType.BOX
        elif (
            isinstance(env.observation_space, GymnasiumDictSpace)
            and "observation" in env.observation_space.spaces
            and "desired_goal" in env.observation_space.spaces
        ):
            env_type = _MinariEnvType.GOAL_CONDITIONED
            unwrapped_env = GoalConcatWrapper(
                unwrapped_env, tuple_observation=tuple_observation
            )
        else:
            raise ValueError(
                f"Unsupported observation space: {env.observation_space}"
            )

        observations = []
        actions = []
        rewards = []
        terminals = []
        timeouts = []

        for ep in _dataset:
            if env_type == _MinariEnvType.BOX:
                _observations = ep.observations
            elif env_type == _MinariEnvType.GOAL_CONDITIONED:
                assert isinstance(ep.observations, dict)
                if isinstance(ep.observations["desired_goal"], dict):
                    sorted_keys = sorted(
                        list(ep.observations["desired_goal"].keys())
                    )
                    goal_obs = np.concatenate(
                        [
                            ep.observations["desired_goal"][key]
                            for key in sorted_keys
                        ],
                        axis=-1,
                    )
                else:
                    goal_obs = ep.observations["desired_goal"]
                if tuple_observation:
                    _observations = (ep.observations["observation"], goal_obs)
                else:
                    _observations = np.concatenate(
                        [
                            ep.observations["observation"],
                            goal_obs,
                        ],
                        axis=-1,
                    )
            else:
                raise ValueError("Unsupported observation format.")
            observations.append(_observations)
            actions.append(ep.actions)
            rewards.append(ep.rewards)
            terminals.append(ep.terminations)
            timeouts.append(ep.truncations)

        if tuple_observation:
            stacked_observations = tuple(
                np.concatenate([observation[i] for observation in observations])
                for i in range(2)
            )
        else:
            stacked_observations = np.concatenate(observations)

        dataset = MDPDataset(
            observations=stacked_observations,
            actions=np.concatenate(actions),
            rewards=np.concatenate(rewards),
            terminals=np.concatenate(terminals),
            timeouts=np.concatenate(timeouts),
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
        )

        return dataset, GymnasiumTimeLimit(
            unwrapped_env, max_episode_steps=env.spec.max_episode_steps
        )

    except ImportError as e:
        raise ImportError(
            "minari is not installed.\n" "$ d3rlpy install minari"
        ) from e


ATARI_GAMES = [
    "adventure",
    "air-raid",
    "alien",
    "amidar",
    "assault",
    "asterix",
    "asteroids",
    "atlantis",
    "bank-heist",
    "battle-zone",
    "beam-rider",
    "berzerk",
    "bowling",
    "boxing",
    "breakout",
    "carnival",
    "centipede",
    "chopper-command",
    "crazy-climber",
    "defender",
    "demon-attack",
    "double-dunk",
    "elevator-action",
    "enduro",
    "fishing-derby",
    "freeway",
    "frostbite",
    "gopher",
    "gravitar",
    "hero",
    "ice-hockey",
    "jamesbond",
    "journey-escape",
    "kangaroo",
    "krull",
    "kung-fu-master",
    "montezuma-revenge",
    "ms-pacman",
    "name-this-game",
    "phoenix",
    "pitfall",
    "pong",
    "pooyan",
    "private-eye",
    "qbert",
    "riverraid",
    "road-runner",
    "robotank",
    "seaquest",
    "skiing",
    "solaris",
    "space-invaders",
    "star-gunner",
    "tennis",
    "time-pilot",
    "tutankham",
    "up-n-down",
    "venture",
    "video-pinball",
    "wizard-of-wor",
    "yars-revenge",
    "zaxxon",
]


def get_dataset(
    env_name: str,
    transition_picker: Optional[TransitionPickerProtocol] = None,
    trajectory_slicer: Optional[TrajectorySlicerProtocol] = None,
    render_mode: Optional[str] = None,
) -> tuple[ReplayBuffer, gym.Env[Any, Any]]:
    """Returns dataset and envrironment by guessing from name.

    This function returns dataset by matching name with the following datasets.

    - cartpole-replay
    - cartpole-random
    - pendulum-replay
    - pendulum-random
    - d4rl-pybullet
    - d4rl-atari
    - d4rl

    .. code-block:: python

       import d3rlpy

       # cartpole dataset
       dataset, env = d3rlpy.datasets.get_dataset('cartpole')

       # pendulum dataset
       dataset, env = d3rlpy.datasets.get_dataset('pendulum')

       # d4rl-atari dataset
       dataset, env = d3rlpy.datasets.get_dataset('breakout-mixed-v0')

       # d4rl dataset
       dataset, env = d3rlpy.datasets.get_dataset('hopper-medium-v0')

    Args:
        env_name: environment id of the dataset.
        transition_picker: TransitionPickerProtocol object.
        trajectory_slicer: TrajectorySlicerProtocol object.
        render_mode: Mode of rendering (``human``, ``rgb_array``).

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    if env_name == "cartpole-replay":
        return get_cartpole(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "cartpole-random":
        return get_cartpole(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "pendulum-replay":
        return get_pendulum(
            dataset_type="replay",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif env_name == "pendulum-random":
        return get_pendulum(
            dataset_type="random",
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif re.match(r"^bullet-.+$", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    elif re.match(r"hopper|halfcheetah|walker|ant", env_name):
        return get_d4rl(
            env_name,
            transition_picker=transition_picker,
            trajectory_slicer=trajectory_slicer,
            render_mode=render_mode,
        )
    raise ValueError(f"Unrecognized env_name: {env_name}.")
