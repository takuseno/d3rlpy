from d3rlpy.algos import DoubleDQN
from d3rlpy.optimizers import AdamFactory
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy
from d4rl_atari.envs import AtariEnv

# get wrapped atari environment
env = AtariEnv('Breakout',
               stack=False,
               clip_reward=True,
               terminate_on_life_loss=True)

eval_env = AtariEnv('Breakout',
                    stack=False,
                    clip_reward=False,
                    terminate_on_life_loss=False)

# setup algorithm
dqn = DoubleDQN(batch_size=32,
                learning_rate=5e-5,
                optim_factory=AdamFactory(eps=1e-2 / 32),
                target_update_interval=40000,
                q_func_factory='iqn',
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
               eval_epsilon=0.01,
               n_steps=50000000,
               n_steps_per_epoch=100000,
               update_interval=4,
               update_start_step=50000)
