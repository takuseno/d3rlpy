import argparse

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="breakout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--pre-stack", action="store_true")
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = d3rlpy.datasets.get_atari_transitions(
        args.game,
        fraction=0.01,
        index=1 if args.game == "asterix" else 0,
        num_stack=4,
        sticky_action=False,
        pre_stack=args.pre_stack,
    )

    d3rlpy.envs.seed_env(env, args.seed)

    if args.game == "pong":
        batch_size = 512
        context_size = 50
    else:
        batch_size = 128
        context_size = 30

    if args.game == "pong":
        target_return = 20
    elif args.game == "breakout":
        target_return = 90
    elif args.game == "qbert":
        target_return = 2500
    elif args.game == "seaquest":
        target_return = 1450
    else:
        raise ValueError(f"target_return is not defined for {args.game}")

    # extract maximum timestep in dataset
    max_timestep = 0
    for episode in dataset.episodes:
        max_timestep = max(max_timestep, episode.transition_count + 1)

    dt = d3rlpy.algos.DiscreteDecisionTransformerConfig(
        batch_size=batch_size,
        context_size=context_size,
        learning_rate=6e-4,
        activation_type="gelu",
        embed_activation_type="tanh",
        encoder_factory=d3rlpy.models.PixelEncoderFactory(
            feature_size=128, exclude_last_activation=True
        ),  # Nature DQN
        num_heads=8,
        num_layers=6,
        attn_dropout=0.1,
        embed_dropout=0.1,
        optim_factory=d3rlpy.models.GPTAdamWFactory(
            betas=(0.9, 0.95),
            weight_decay=0.1,
        ),
        clip_grad_norm=1.0,
        warmup_tokens=512 * 20,
        final_tokens=2 * 500000 * context_size * 3,
        observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),
        max_timestep=max_timestep,
    ).create(device=args.gpu)

    n_steps_per_epoch = dataset.transition_count // batch_size
    n_steps = n_steps_per_epoch * 5
    dt.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        eval_env=env,
        eval_target_return=target_return,
        eval_action_sampler=d3rlpy.algos.SoftmaxTransformerActionSampler(
            temperature=1.0,
        ),
        experiment_name=f"DiscreteDT_{args.game}_{args.seed}",
    )


if __name__ == "__main__":
    main()
