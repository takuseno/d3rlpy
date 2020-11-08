from d3rlpy.algos import DiscreteSAC
from d3rlpy.online.buffers import ReplayBuffer
from d4rl_atari.envs import AtariEnv

# get wrapped atari environment
env = AtariEnv('Breakout', stack=False, clip_reward=True)
eval_env = AtariEnv('Breakout', stack=False, clip_reward=False)

# setup algorithm
sac = DiscreteSAC(target_update_interval=8000,
                  scaler='pixel',
                  n_frames=4,
                  use_gpu=True)

# replay buffer for experience replay
buffer = ReplayBuffer(maxlen=1000000, env=env)

# start training
sac.fit_online(env,
               buffer,
               eval_env=eval_env,
               eval_epsilon=0.01,
               n_steps=1000000,
               n_steps_per_epoch=10000,
               update_interval=4,
               update_start_step=20000)
