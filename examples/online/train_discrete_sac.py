from d3rlpy.algos import DiscreteSAC
from d3rlpy.online.buffers import ReplayBuffer
from d4rl_atari.envs import AtariEnv

# get wrapped atari environment
env = AtariEnv('Breakout', stack=False, clip_reward=True)
eval_env = AtariEnv('Breakout', stack=False, clip_reward=False)

# setup algorithm
sac = DiscreteSAC(target_update_interval=8000 / 4,
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
               eval_interval=100,
               n_epochs=50000,
               n_steps_per_epoch=1000,
               n_updates_per_epoch=250,
               update_start_step=20000,
               save_interval=1000)
