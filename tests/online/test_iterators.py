import pytest
import gym

from d3rlpy.algos import DQN, SAC
from d3rlpy.online.iterators import train
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy


def test_train_with_dqn():
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')

    algo = DQN(n_epochs=1)

    buffer = ReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    train(env,
          algo,
          buffer,
          explorer,
          eval_env=eval_env,
          logdir='test_data',
          tensorboard=False)


def test_train_with_sac():
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')

    algo = SAC(n_epochs=1)

    buffer = ReplayBuffer(1000, env)

    train(env,
          algo,
          buffer,
          eval_env=eval_env,
          logdir='test_data',
          tensorboard=False)
