import argparse
import d3rlpy

from d3rlpy.algos import CQL
from d3rlpy.datasets import get_pybullet
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.metrics.scorer import continuous_action_diff_scorer
from d3rlpy.metrics.scorer import value_estimation_std_scorer
from d3rlpy.gpu import Device
from sklearn.model_selection import train_test_split


def main(args):
    dataset, env = get_pybullet(args.dataset, as_tensor=True, device=args.gpu)

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    device = None if args.gpu is None else Device(args.gpu)

    cql = CQL(n_epochs=100, q_func_type=args.q_func_type, use_gpu=device)

    cql.fit(train_episodes,
            eval_episodes=test_episodes,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer,
                'value_scale': average_value_estimation_scorer,
                'value_std': value_estimation_std_scorer,
                'action_diff': continuous_action_diff_scorer
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='hopper-bullet-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--q-func-type',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args)
