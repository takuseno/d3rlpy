import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    cql = d3rlpy.algos.CQL(actor_encoder_factory=encoder,
                           critic_encoder_factory=encoder,
                           alpha_learning_rate=0.0,
                           use_gpu=args.gpu)

    scorers = {
        'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env),
        'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer
    }

    cql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=500000,
            n_steps_per_epoch=1000,
            save_interval=10,
            scorers=scorers,
            experiment_name=f"CQL_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()
