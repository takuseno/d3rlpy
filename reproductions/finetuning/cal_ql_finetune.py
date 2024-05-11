import argparse
import math

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="antmaze-medium-diverse-v2"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    # sparse reward setup requires special treatment for failure trajectories
    transition_picker = d3rlpy.dataset.SparseRewardTransitionPicker(
        failure_return=-49.5,  # ((-5 / 0.01) + 5) / 10
        step_reward=0,
    )

    dataset, env = d3rlpy.datasets.get_d4rl(
        args.dataset,
        transition_picker=transition_picker,
    )

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # for antmaze datasets
    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(
        shift=-5,
        multiplier=10.0,
        multiply_first=True,
    )

    encoder = d3rlpy.models.encoders.VectorEncoderFactory(
        [256, 256, 256, 256],
    )

    cal_ql = d3rlpy.algos.CalQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        alpha_learning_rate=3e-4,
        initial_alpha=math.e,
        batch_size=256,
        conservative_weight=5.0,
        critic_encoder_factory=encoder,
        alpha_threshold=0.8,
        reward_scaler=reward_scaler,
        max_q_backup=True,
    ).create(device=args.gpu)

    # pretraining
    cal_ql.fit(
        dataset,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CalQL_pretraining_{args.dataset}_{args.seed}",
    )

    # prepare FIFO buffer filled with dataset episodes
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=1000000,
        env=env,
        transition_picker=transition_picker,
    )

    # sample half from offline dataset and the rest from online buffer
    mixed_buffer = d3rlpy.dataset.MixedReplayBuffer(
        primary_replay_buffer=buffer,
        secondary_replay_buffer=dataset,
        secondary_mix_ratio=0.5,
    )

    # finetuning
    _, eval_env = d3rlpy.datasets.get_d4rl(args.dataset)
    d3rlpy.envs.seed_env(eval_env, args.seed)
    cal_ql.fit_online(
        env,
        buffer=mixed_buffer,
        eval_env=eval_env,
        experiment_name=f"CalQL_finetuning_{args.dataset}_{args.seed}",
        n_steps=1000000,
        n_steps_per_epoch=1000,
        n_updates=1000,
        update_interval=1000,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
