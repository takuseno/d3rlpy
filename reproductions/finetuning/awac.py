import argparse

import gym
from sklearn.model_selection import train_test_split

import d3rlpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="antmaze-umaze-v0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256, 256])
    optim = d3rlpy.models.optimizers.AdamFactory(weight_decay=1e-4)
    # for antmaze datasets
    reward_scaler = d3rlpy.preprocessing.ConstantShiftRewardScaler(shift=-1)

    awac = d3rlpy.algos.AWAC(
        actor_learning_rate=3e-4,
        actor_encoder_factory=encoder,
        actor_optim_factory=optim,
        critic_learning_rate=3e-4,
        batch_size=1024,
        lam=1.0,
        reward_scaler=reward_scaler,
        use_gpu=args.gpu,
    )

    awac.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_steps=25000,
        n_steps_per_epoch=5000,
        save_interval=10,
        scorers={
            "environment": d3rlpy.metrics.evaluate_on_environment(env),
            "value_scale": d3rlpy.metrics.average_value_estimation_scorer,
        },
        experiment_name=f"AWAC_pretraining_{args.dataset}_{args.seed}",
    )

    # finetuning
    buffer = d3rlpy.online.buffers.ReplayBuffer(
        maxlen=1000000, episodes=dataset.episodes
    )
    eval_env = gym.make(env.unwrapped.spec.id)
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
