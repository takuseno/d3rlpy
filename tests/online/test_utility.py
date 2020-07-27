import gym

from d3rlpy.online.utility import get_action_size_from_env


def test_get_action_size_from_env_with_cartpole():
    env = gym.make('CartPole-v0')

    action_size = get_action_size_from_env(env)

    assert action_size == env.action_space.n


def test_get_action_size_from_env_with_pendulum():
    env = gym.make('Pendulum-v0')

    action_size = get_action_size_from_env(env)

    assert action_size == env.action_space.shape[0]
