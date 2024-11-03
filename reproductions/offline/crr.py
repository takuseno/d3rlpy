import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hopper-medium-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    crr = d3rlpy.algos.CRRConfig(
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        batch_size=256,
        weight_type="binary",
        advantage_type="mean",
        target_update_type="soft",
        compile_graph=args.compile,
    ).create(device=args.gpu)

    crr.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CRR_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
