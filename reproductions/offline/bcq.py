import argparse
import d3rlpy
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hopper-medium-v0')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    d3rlpy.seed(args.seed)

    dataset, env = d3rlpy.datasets.get_d4rl(args.gpu)

    _, test_episodes = train_test_split(dataset, test_size=0.2)

    vae_encoder = d3rlpy.models.encoders.VectorEncoderFactory([750, 750])

    rl_encoder = d3rlpy.models.encoders.VectorEncoderFactory([400, 300])

    bcq = d3rlpy.algos.BCQ(actor_encoder_factory=rl_encoder,
                           critic_encoder_factory=rl_encoder,
                           imitator_encoder_factory=vae_encoder,
                           use_gpu=args.gpu)

    scorers = {
        'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env),
        'value_scale': d3rlpy.metrics.scorer.average_value_estimation_scorer
    }

    bcq.fit(dataset.episodes,
            eval_episodes=test_episodes,
            n_epochs=2000,
            scorers=scorers)


if __name__ == '__main__':
    main()
