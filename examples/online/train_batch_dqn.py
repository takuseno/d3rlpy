import gym

from d3rlpy.algos import DQN
from d3rlpy.envs import BatchEnvWrapper
from d3rlpy.online.buffers import BatchReplayBuffer
from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

if __name__ == '__main__':
    env = BatchEnvWrapper([lambda: gym.make('CartPole-v0') for _ in range(10)])
    eval_env = gym.make('CartPole-v0')

    # setup algorithm
    dqn = DQN(batch_size=32,
              learning_rate=1e-3,
              target_update_interval=1000,
              use_gpu=False)

    # replay buffer for experience replay
    buffer = BatchReplayBuffer(maxlen=100000, env=env)

    # epilon-greedy explorer
    explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                        end_epsilon=0.1,
                                        duration=100000)

    # start training
    dqn.fit_batch_online(env,
                         buffer,
                         explorer,
                         n_epochs=100,
                         eval_interval=1,
                         eval_env=eval_env,
                         n_steps_per_epoch=1000,
                         n_updates_per_epoch=1000)
