import gym

from d3rlpy.algos import DQN
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.online.iterators import train


env = gym.make('CartPole-v0')
eval_env = gym.make('CartPole-v0')

# setup algorithm
dqn = DQN(n_epochs=30,
          batch_size=32,
          learning_rate=2.5e-4,
          target_update_interval=100,
          use_gpu=False)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=100000, env=env)

# epilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.1,
                                    duration=10000)

# start training
train(env,
      dqn,
      buffer,
      explorer,
      eval_env=eval_env,
      n_steps_per_epoch=1000,
      n_updates_per_epoch=100)
