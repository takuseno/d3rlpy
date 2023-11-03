import gym
import gymnasium
import numpy as np
import pytest
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Dict as GymnasiumDictSpace

from d3rlpy.algos import DQNConfig
from d3rlpy.envs.wrappers import (
    Atari,
    ChannelFirst,
    FrameStack,
    GoalConcatWrapper,
)

from ..dummy_env import DummyAtari


def test_channel_first() -> None:
    env = DummyAtari(grayscale=False)

    assert env.observation_space.shape
    width, height, channel = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (channel, width, height)  # type: ignore

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (channel, width, height)  # type: ignore

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


def test_channel_first_with_2_dim_obs() -> None:
    env = DummyAtari(squeeze=True)

    assert env.observation_space.shape
    width, height = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (1, width, height)  # type: ignore

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (1, width, height)  # type: ignore

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


@pytest.mark.parametrize("num_stack", [4])
def test_frame_stack(num_stack: int) -> None:
    env = DummyAtari(squeeze=True)

    assert env.observation_space.shape
    width, height = env.observation_space.shape

    wrapper = FrameStack(env, num_stack=num_stack)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (num_stack, width, height)  # type: ignore

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (num_stack, width, height)  # type: ignore

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


@pytest.mark.skip(reason="This needs actual Atari 2600 environments.")
@pytest.mark.parametrize("is_eval", [True])
def test_atari(is_eval: bool) -> None:
    env = Atari(gym.make("BreakoutNoFrameskip-v4"), is_eval)

    assert env.observation_space.shape == (1, 84, 84)

    # check reset
    observation, _ = env.reset()
    assert observation.shape == (1, 84, 84)  # type: ignore

    # check step
    observation, _, _, _, _ = env.step(env.action_space.sample())
    assert observation.shape == (1, 84, 84)  # type: ignore


def test_goal_concat_wrapper() -> None:
    raw_env = gymnasium.make("AntMaze_UMaze-v4")
    env = GoalConcatWrapper(raw_env)

    assert isinstance(raw_env.observation_space, GymnasiumDictSpace)

    observation_space = raw_env.observation_space["observation"]
    assert isinstance(observation_space, GymnasiumBox)
    observation_shape = observation_space.shape

    goal_space = raw_env.observation_space["desired_goal"]
    assert isinstance(goal_space, GymnasiumBox)
    goal_shape = goal_space.shape

    concat_shape = (observation_shape[0] + goal_shape[0],)

    # check reset
    observation, _ = env.reset()
    assert observation.shape == concat_shape  # type: ignore

    # check step
    observation, _, _, _, _ = env.step(env.action_space.sample())
    assert observation.shape == concat_shape  # type: ignore
