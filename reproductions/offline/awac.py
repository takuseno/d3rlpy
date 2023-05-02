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

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256, 256])
    optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=1e-4)

    awac = d3rlpy.algos.AWACConfig(
        actor_learning_rate=3e-4,
        actor_encoder_factory=encoder,
        actor_optim_factory=optim,
        critic_learning_rate=3e-4,
        critic_encoder_factory=encoder,
        batch_size=1024,
        lam=1.0,
    ).create(args.gpu)

    awac.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"AWAC_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
