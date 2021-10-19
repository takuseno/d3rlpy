import argparse
import d3rlpy
from sklearn.model_selection import train_test_split
from d3rlpy.models.optimizers import AdamFactory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='walker2d-bullet-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    iql = d3rlpy.algos.IQL(beta=3.0,
                           expectile=0.7,
                           max_weight=100.0,
                           actor_learning_rate=3e-4,
                           critic_learning_rate=3e-4,
                           actor_encoder_factory=encoder,
                           actor_optim_factory=AdamFactory(weight_decay=0.0),
                           critic_optim_factory=AdamFactory(weight_decay=0.0),
                           critic_encoder_factory=encoder,
                           use_gpu=args.gpu)

    iql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=10000,
            save_interval=10,
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
                'initial_state_value': d3rlpy.metrics.initial_state_value_estimation_scorer,
            },
            experiment_name=f"IQL_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()
