import argparse
import copy

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="antmaze-umaze-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_minari(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    # for antmaze datasets
    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1)

    cal_ql = d3rlpy.algos.CalQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        alpha_threshold=0.8,
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    # pretraining
    cal_ql.fit(
        dataset,
        n_steps=1000000,
        n_steps_per_epoch=100000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CalQL_pretraining_{args.dataset}_{args.seed}",
    )

    # prepare FIFO buffer filled with dataset episodes
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(1000000)

    mixed_buffer = d3rlpy.dataset.MixedReplayBuffer(
        primary_replay_buffer=buffer,
        secondary_replay_buffer=dataset,
        secondary_mix_ratio=0.5,
    )

    # finetuning
    eval_env = copy.deepcopy(env)
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
