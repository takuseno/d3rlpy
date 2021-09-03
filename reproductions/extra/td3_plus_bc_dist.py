import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--q-func', type=str, required=True)
    args = parser.parse_args()

    dataset, env = d3rlpy.datasets.get_dataset(args.dataset)

    # fix seed
    d3rlpy.seed(args.seed)
    env.seed(args.seed)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    optim = d3rlpy.models.optimizers.AdamFactory(eps=1e-2 / 256)

    td3 = d3rlpy.algos.TD3PlusBC(q_func_factory=args.q_func,
                                 critic_optim_factory=optim,
                                 use_gpu=args.gpu)

    td3.fit(
        dataset.episodes,
        eval_episodes=test_episodes,
        n_steps=500000,
        n_steps_per_epoch=1000,
        save_interval=10,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'value_scale': d3rlpy.metrics.average_value_estimation_scorer,
        },
        experiment_name=f"TD3PlusBC_{args.q_func}_{args.dataset}_{args.seed}")


if __name__ == '__main__':
    main()
