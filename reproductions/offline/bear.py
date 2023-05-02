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

    vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])

    if "halfcheetah" in env.unwrapped.spec.id.lower():
        kernel = "gaussian"
    else:
        kernel = "laplacian"

    bear = d3rlpy.algos.BEARConfig(
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
        imitator_learning_rate=3e-4,
        alpha_learning_rate=1e-3,
        imitator_encoder_factory=vae_encoder,
        temp_learning_rate=0.0,
        initial_temperature=1e-20,
        batch_size=256,
        mmd_sigma=20.0,
        mmd_kernel=kernel,
        n_mmd_action_samples=4,
        alpha_threshold=0.05,
        n_target_samples=10,
        n_action_samples=100,
        warmup_steps=40000,
    ).create(device=args.gpu)

    bear.fit(
        dataset,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"BEAR_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
