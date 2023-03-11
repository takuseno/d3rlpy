import gym

from d3rlpy.algos import SAC
from d3rlpy.envs import AsyncBatchEnv
from d3rlpy.online.buffers import BatchReplayBuffer

if __name__ == "__main__":
    env = AsyncBatchEnv([lambda: gym.make("Pendulum-v0") for _ in range(10)])
    eval_env = gym.make("Pendulum-v0")

    # setup algorithm
    sac = SAC(batch_size=100, use_gpu=False)

    # replay buffer for experience replay
    buffer = BatchReplayBuffer(maxlen=100000, env=env)

    # start training
    sac.fit_batch_online(
        env,
        buffer,
        n_epochs=100,
        eval_interval=1,
        eval_env=eval_env,
        n_steps_per_epoch=1000,
        n_updates_per_epoch=1000,
    )
