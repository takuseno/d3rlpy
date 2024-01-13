import argparse

import d3rlpy

ACTION_FLEXIBILITY = {
    "walker2d-random-v0": 0.05,
    "hopper-random-v0": 0.5,
    "halfcheetah-random-v0": 0.01,
    "walker2d-medium-v0": 0.1,
    "hopper-medium-v0": 0.01,
    "halfcheetah-medium-v0": 0.1,
    "walker2d-medium-relay-v0": 0.01,
    "hopper-medium-replay-v0": 0.2,
    "halfcheetah-medium-replay-v0": 0.05,
    "walker2d-medium-expert-v0": 0.01,
    "hopper-medium-expert-v0": 0.01,
    "halfcheetah-medium-expert-v0": 0.01,
}


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

    if "medium-replay" in args.dataset:
        vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([128, 128])
    else:
        vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])

    plas = d3rlpy.algos.PLASWithPerturbationConfig(
        actor_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_learning_rate=1e-3,
        critic_encoder_factory=encoder,
        imitator_learning_rate=1e-4,
        imitator_encoder_factory=vae_encoder,
        batch_size=100,
        lam=1.0,
        warmup_steps=500000,
        action_flexibility=ACTION_FLEXIBILITY[args.dataset],
    ).create(device=args.gpu)

    plas.fit(
        dataset,
        n_steps=1000000,  # RL starts at 500000 step
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"PLASWithPerturbation_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
