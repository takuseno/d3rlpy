import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = d3rlpy.datasets.get_atari(args.dataset)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    cql = d3rlpy.algos.DiscreteCQL(
        optim_factory=d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 32),
        scaler='pixel',
        n_frames=4,
        q_func_factory='qr',
        use_gpu=args.gpu)

    env_scorer = d3rlpy.metrics.evaluate_on_environment(env, epsilon=0.001)

    cql.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_steps=50000000,
            n_steps_per_epoch=10000,
            scorers={
                'environment': env_scorer,
                'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
            },
            experiment_name=f"DiscreteCQL_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()
