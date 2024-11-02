import argparse
import math

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

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256])

    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    cql = d3rlpy.algos.CQLConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=1e-4,
        alpha_learning_rate=3e-4,
        initial_alpha=math.e,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        batch_size=256,
        n_action_samples=10,
        alpha_threshold=10,
        conservative_weight=conservative_weight,
    ).create(device=args.gpu)

    cql.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"CQL_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
