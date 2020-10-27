import argparse
import d3rlpy

from d3rlpy.algos import DiscreteBC
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.gpu import Device
from sklearn.model_selection import train_test_split


def main(args):
    dataset, env = get_atari(args.dataset)

    d3rlpy.seed(args.seed)

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    bc = DiscreteBC(
        n_frames=4,  # frame stacking
        scaler='pixel',
        use_gpu=args.gpu)

    bc.fit(train_episodes,
           eval_episodes=test_episodes,
           n_epochs=100,
           scorers={'environment': evaluate_on_environment(env, epsilon=0.05)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='breakout-mixed-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    main(args)
