import argparse

import gym

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    # get wrapped atari environment
    env = d3rlpy.envs.Atari(gym.make(args.env), num_stack=4)
    eval_env = d3rlpy.envs.Atari(gym.make(args.env), num_stack=4, is_eval=True)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)
    d3rlpy.envs.seed_env(eval_env, args.seed)

    # setup algorithm
    dqn = d3rlpy.algos.DQNConfig(
        batch_size=32,
        learning_rate=5e-5,
        optim_factory=d3rlpy.optimizers.AdamFactory(eps=1e-2 / 32),
        target_update_interval=10000 // 4,
        q_func_factory=d3rlpy.models.q_functions.QRQFunctionFactory(
            n_quantiles=200
        ),
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        compile_graph=args.compile,
    ).create(device=args.gpu)

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=1000000,
        transition_picker=d3rlpy.dataset.FrameStackTransitionPicker(n_frames=4),
        writer_preprocessor=d3rlpy.dataset.LastFrameWriterPreprocess(),
        env=env,
    )

    # epilon-greedy explorer
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=1.0, end_epsilon=0.01, duration=1000000
    )

    # start training
    dqn.fit_online(
        env,
        buffer,
        explorer,
        eval_env=eval_env,
        eval_epsilon=0.001,
        n_steps=50000000,
        n_steps_per_epoch=100000,
        update_interval=4,
        update_start_step=50000,
    )


if __name__ == "__main__":
    main()
