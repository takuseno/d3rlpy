import argparse
import gym
import d3rlpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    # get wrapped atari environment
    env = d3rlpy.envs.Atari(gym.make(args.env))
    eval_env = d3rlpy.envs.Atari(gym.make(args.env), is_eval=True)

    # setup algorithm
    dqn = d3rlpy.algos.DQN(
        batch_size=32,
        learning_rate=5e-5,
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        target_update_interval=10000 // 4,
        q_func_factory='fqf',
        scaler='pixel',
        n_frames=4,
        use_gpu=True)

    # replay buffer for experience replay
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

    # epilon-greedy explorer
    explorer = d3rlpy.online.explorers.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.01, duration=1000000)

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


if __name__ == '__main__':
    main()
