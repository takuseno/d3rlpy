import argparse
import d3rlpy

from d3rlpy.algos import DQN
from d3rlpy.datasets import get_atari
from d3rlpy.preprocessing import PixelScaler
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from sklearn.model_selection import train_test_split


def main(args):
    dataset, env = get_atari(args.dataset)

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    dqn = DQN(n_epochs=100,
              q_func_type=args.q_func_type,
              scaler=PixelScaler(),
              use_gpu=args.gpu)

    dqn.fit(train_episodes,
            eval_episodes=test_episodes,
            scorers={
                'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer,
                'discounted_advantage': discounted_sum_of_advantage_scorer
            })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--q-func-type',
                        type=str,
                        default='mean',
                        choices=['mean', 'qr', 'iqn', 'fqf'])
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    main(args)
