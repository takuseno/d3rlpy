import argparse

import d3rlpy

BETA_TABLE: dict[str, tuple[float, float]] = {
    "halfcheetah-random": (0.001, 0.1),
    "halfcheetah-medium": (0.001, 0.01),
    "halfcheetah-expert": (0.01, 0.01),
    "halfcheetah-medium-replay": (0.01, 0.001),
    "halfcheetah-full-replay": (0.001, 0.1),
    "hopper-random": (0.001, 0.01),
    "hopper-medium": (0.01, 0.001),
    "hopper-expert": (0.1, 0.001),
    "hopper-medium-expert": (0.1, 0.01),
    "hopper-medium-replay": (0.05, 0.5),
    "hopper-full-replay": (0.01, 0.01),
    "walker2d-random": (0.01, 0.0),
    "walker2d-medium": (0.05, 0.1),
    "walker2d-expert": (0.01, 0.5),
    "walker2d-medium-expert": (0.01, 0.01),
    "walker2d-medium-replay": (0.05, 0.01),
    "walker2d-full-replay": (0.01, 0.01),
}


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

    # deeper network
    actor_encoder = d3rlpy.models.VectorEncoderFactory([256, 256, 256])
    critic_encoder = d3rlpy.models.VectorEncoderFactory(
        [256, 256, 256], use_layer_norm=True
    )

    actor_beta, critic_beta = 0.01, 0.01
    for dataset_name, beta_from_paper in BETA_TABLE.items():
        if dataset_name in args.dataset:
            actor_beta, critic_beta = beta_from_paper
            break

    rebrac = d3rlpy.algos.ReBRACConfig(
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        batch_size=1024,
        gamma=0.99,
        actor_encoder_factory=actor_encoder,
        critic_encoder_factory=critic_encoder,
        target_smoothing_sigma=0.2,
        target_smoothing_clip=0.5,
        update_actor_interval=2,
        actor_beta=actor_beta,
        critic_beta=critic_beta,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        compile_graph=args.compile,
    ).create(device=args.gpu)

    rebrac.fit(
        dataset,
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"ReBRAC_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
