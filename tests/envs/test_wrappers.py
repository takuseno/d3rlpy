import gym
import numpy as np
import pytest

from d3rlpy.algos import DQNConfig
from d3rlpy.envs.wrappers import Atari, ChannelFirst, FrameStack

from ..dummy_env import DummyAtari


def test_channel_first():
    env = DummyAtari(grayscale=False)

    width, height, channel = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (channel, width, height)

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (channel, width, height)

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


def test_channel_first_with_2_dim_obs():
    env = DummyAtari(squeeze=True)

    width, height = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (1, width, height)

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (1, width, height)

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


@pytest.mark.parametrize("num_stack", [4])
def test_frame_stack(num_stack):
    env = DummyAtari(squeeze=True)

    width, height = env.observation_space.shape

    wrapper = FrameStack(env, num_stack=num_stack)

    # check reset
    observation, _ = wrapper.reset()
    assert observation.shape == (num_stack, width, height)

    # check step
    observation, _, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (num_stack, width, height)

    # check with algorithm
    dqn = DQNConfig().create()
    dqn.build_with_env(wrapper)
    dqn.predict(np.expand_dims(observation, axis=0))


@pytest.mark.skip(reason="This needs actual Atari 2600 environments.")
@pytest.mark.parametrize("is_eval", [100])
def test_atari(is_eval):
    env = Atari(gym.make("BreakoutNoFrameskip-v4"), is_eval)

    assert env.observation_space.shape == (1, 84, 84)

    # check reset
    observation, _ = env.reset()
    assert observation.shape == (1, 84, 84)

    # check step
    observation, _, _, _, _ = env.step(env.action_space.sample())
    assert observation.shape == (1, 84, 84)
