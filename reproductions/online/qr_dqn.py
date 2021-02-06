import gym

from gym.wrappers import AtariPreprocessing, TransformReward
from d3rlpy.algos import DQN
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.models.q_functions import QRQFunctionFactory
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d3rlpy.envs import Atari

# get wrapped atari environment
env = Atari(gym.make('BreakoutNoFrameskip-v4'))
eval_env = Atari(gym.make('BreakoutNoFrameskip-v4'), is_eval=True)

# setup algorithm
dqn = DQN(batch_size=32,
          learning_rate=5e-5,
          optim_factory=AdamFactory(eps=1e-2 / 32),
          target_update_interval=10000 // 4,
          q_func_factory=QRQFunctionFactory(n_quantiles=200),
          scaler='pixel',
          n_frames=4,
          use_gpu=True)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=1000000, env=env)

# epilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.01,
                                    duration=1000000)

# start training
dqn.fit_online(env,
               buffer,
               explorer,
               eval_env=eval_env,
               eval_epsilon=0.001,
               n_steps=50000000,
               n_steps_per_epoch=100000,
               update_interval=4,
               update_start_step=50000)
