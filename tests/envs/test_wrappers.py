import gym

from gym.wrappers import AtariPreprocessing
from d3rlpy.algos import DQN
from d3rlpy.envs.wrappers import ChannelFirst


def test_channel_first():
    env = gym.make("Breakout-v0")

    width, height, channel = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation = wrapper.reset()
    assert observation.shape == (channel, width, height)

    # check step
    observation, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (channel, width, height)

    # check with algorithm
    dqn = DQN()
    dqn.build_with_env(wrapper)
    dqn.predict([observation])


def test_channel_first_with_2_dim_obs():
    env = AtariPreprocessing(gym.make("BreakoutNoFrameskip-v4"))

    width, height = env.observation_space.shape

    wrapper = ChannelFirst(env)

    # check reset
    observation = wrapper.reset()
    assert observation.shape == (1, width, height)

    # check step
    observation, _, _, _ = wrapper.step(wrapper.action_space.sample())
    assert observation.shape == (1, width, height)

    # check with algorithm
    dqn = DQN()
    dqn.build_with_env(wrapper)
    dqn.predict([observation])
