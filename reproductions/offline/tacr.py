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

    if "halfcheetah" in args.dataset:
        target_return = 6000
    elif "hopper" in args.dataset:
        target_return = 3600
    elif "walker" in args.dataset:
        target_return = 5000
    else:
        raise ValueError("unsupported dataset")

    tacr = d3rlpy.algos.TACRConfig(
        batch_size=64,
        learning_rate=1e-4,
        optim_factory=d3rlpy.optimizers.AdamWFactory(
            weight_decay=1e-4,
            clip_grad_norm=0.25,
            lr_scheduler_factory=d3rlpy.optimizers.WarmupSchedulerFactory(
                warmup_steps=10000
            ),
        ),
        encoder_factory=d3rlpy.models.VectorEncoderFactory(
            [128],
            exclude_last_activation=True,
        ),
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(),
        reward_scaler=d3rlpy.preprocessing.MultiplyRewardScaler(0.001),
        position_encoding_type=d3rlpy.PositionEncodingType.SIMPLE,
        context_size=20,
        num_heads=1,
        num_layers=3,
        max_timestep=1000,
        compile_graph=args.compile,
    ).create(device=args.gpu)

    tacr.fit(
        dataset,
        n_steps=100000,
        n_steps_per_epoch=1000,
        save_interval=10,
        eval_env=env,
        eval_target_return=target_return,
        experiment_name=f"TACR_{args.dataset}_{args.seed}",
    )


if __name__ == "__main__":
    main()
