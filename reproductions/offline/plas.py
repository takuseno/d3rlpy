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

    if "medium-replay" in env.unwrapped.spec.id.lower():
        vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([128, 128])
    else:
        vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])

    plas = d3rlpy.algos.PLASConfig(
        actor_learning_rate=1e-4,
        actor_encoder_factory=encoder,
        critic_learning_rate=1e-3,
        critic_encoder_factory=encoder,
        imitator_learning_rate=1e-4,
        imitator_encoder_factory=vae_encoder,
        batch_size=100,
        lam=1.0,
        warmup_steps=500000,
    ).create(device=args.gpu)

    plas.fit(
        dataset,
        n_steps=1000000,  # RL starts at 500000 step
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"PLAS_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
