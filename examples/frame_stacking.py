import argparse

import gym

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    # get wrapped atari environment with 4 frame stacking
    # observation shape is [4, 84, 84]
    env = d3rlpy.envs.Atari(gym.make(args.env), num_stack=4)
    eval_env = d3rlpy.envs.Atari(gym.make(args.env), num_stack=4, is_eval=True)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    dqn = d3rlpy.algos.DQNConfig(
        batch_size=32,
        learning_rate=2.5e-4,
        optim_factory=d3rlpy.optimizers.RMSpropFactory(),
        target_update_interval=10000 // 4,
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
    ).create(device=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=1000000,
        # stack last 4 frames (stacked shape is [4, 84, 84])
        transition_picker=d3rlpy.dataset.FrameStackTransitionPicker(n_frames=4),
        # store only last frame to save memory (stored shape is [1, 84, 84])
        writer_preprocessor=d3rlpy.dataset.LastFrameWriterPreprocess(),
        env=env,
    )

    # epilon-greedy explorer
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.1, duration=1000000
    )

    # start training
    dqn.fit_online(
        env,
        buffer,
        explorer,
        eval_env=eval_env,
        eval_epsilon=0.01,
        n_steps=1000000,
        n_steps_per_epoch=100000,
        update_interval=4,
        update_start_step=50000,
    )


if __name__ == "__main__":
    main()
