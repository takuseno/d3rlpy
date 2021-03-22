import argparse
import gym
import d3rlpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v2')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    # setup algorithm
    sac = d3rlpy.algos.SAC(batch_size=256,
                           actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           temp_learning_rate=3e-4,
                           use_gpu=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=1000000, env=env)

    # start training
    sac.fit_online(env,
                   buffer,
                   eval_env=eval_env,
                   n_steps=1000000,
                   n_steps_per_epoch=10000,
                   update_interval=1,
                   update_start_step=1000)


if __name__ == '__main__':
    main()
