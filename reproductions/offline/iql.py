import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
        multiplier=1000.0
    )

    iql = d3rlpy.algos.IQLConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        actor_optim_factory=d3rlpy.models.AdamFactory(
            lr_scheduler_factory=d3rlpy.models.CosineAnnealingLRFactory(
                T_max=500000
            ),
        ),
        batch_size=256,
        weight_temp=3.0,
        max_weight=100.0,
        expectile=0.7,
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    iql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"IQL_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
