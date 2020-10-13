import copy

from d3rlpy.algos import DoubleDQN
from d3rlpy.datasets import get_atari
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

# get wrapped atari environment
_, env = get_atari('breakout-mixed-v0')
eval_env = copy.deepcopy(env)

# setup algorithm
dqn = DoubleDQN(n_epochs=10000,
                batch_size=32,
                learning_rate=2.5e-4,
                target_update_interval=10000,
                q_func_type='mean',
                use_gpu=True)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=1000000, env=env)

# epilon-greedy explorer
explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                    end_epsilon=0.1,
                                    duration=1000000)

# start training
dqn.fit_online(env,
               buffer,
               explorer,
               eval_env=eval_env,
               eval_epsilon=0.01,
               n_steps_per_epoch=1000,
               n_updates_per_epoch=1000,
               update_start_step=50000)
