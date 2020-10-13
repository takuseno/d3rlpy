import pytest
import gym

from d3rlpy.algos import DQN, SAC
from d3rlpy.online.iterators import train
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy


def test_train_cartpole_with_dqn():
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


def test_fit_online_cartpole_with_dqn():
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')

    algo = DQN(n_epochs=1)

    buffer = ReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    algo.fit_online(env,
                    buffer,
                    explorer,
                    eval_env=eval_env,
                    logdir='test_data',
                    tensorboard=False)


def test_train_atari_with_dqn():
    import d4rl_atari
    env = gym.make('breakout-mixed-v0', stack=False)
    eval_env = gym.make('breakout-mixed-v0', stack=False)

    algo = DQN(n_epochs=1, n_frames=4)

    buffer = ReplayBuffer(1000, env)

    explorer = LinearDecayEpsilonGreedy()

    train(env,
          algo,
          buffer,
          explorer,
          eval_env=eval_env,
          n_steps_per_epoch=1000,
          n_updates_per_epoch=1,
          logdir='test_data',
          tensorboard=False)

    assert algo.impl.observation_shape == (4, 84, 84)


def test_train_pendulum_with_sac():
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


def test_fit_online_pendulum_with_sac():
    env = gym.make('Pendulum-v0')
    eval_env = gym.make('Pendulum-v0')

    algo = SAC(n_epochs=1)

    buffer = ReplayBuffer(1000, env)

    algo.fit_online(env,
                    buffer,
                    eval_env=eval_env,
                    logdir='test_data',
                    tensorboard=False)
