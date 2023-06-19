import argparse
import copy

import d3rlpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="antmaze-umaze-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    d3rlpy.envs.seed_env(env, args.seed)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256, 256])
    optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=1e-4)
    # for antmaze datasets
    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1)

    awac = d3rlpy.algos.AWACConfig(
        actor_learning_rate=3e-4,
        actor_encoder_factory=encoder,
        actor_optim_factory=optim,
        critic_learning_rate=3e-4,
        batch_size=1024,
        lam=1.0,
        reward_scaler=reward_scaler,
    ).create(device=args.gpu)

    awac.fit(
        dataset,
        n_steps=25000,
        n_steps_per_epoch=5000,
        save_interval=10,
        evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env)},
        experiment_name=f"AWAC_pretraining_{args.dataset}_{args.seed}",
    )

    # prepare FIFO buffer filled with dataset episodes
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(1000000)
    for episode in dataset.episodes:
        buffer.append_episode(episode)

    # finetuning
    eval_env = copy.deepcopy(env)
    d3rlpy.envs.seed_env(eval_env, args.seed)
    awac.fit_online(
        env,
        buffer=buffer,
        eval_env=eval_env,
        experiment_name=f"AWAC_finetuning_{args.dataset}_{args.seed}",
        n_steps=1000000,
        n_steps_per_epoch=1000,
        save_interval=10,
    )


if __name__ == "__main__":
    main()
