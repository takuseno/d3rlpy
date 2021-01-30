from d3rlpy.algos import DiscreteCQL
from d3rlpy.datasets import get_atari
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

dataset, env = get_atari('breakout-medium-v0')

_, test_episodes = train_test_split(dataset, test_size=0.2)

cql = DiscreteCQL(scaler='pixel',
                  n_frames=4,
                  q_func_factory='qr',
                  use_gpu=True)

cql.fit(dataset.episodes,
        eval_episodes=test_episodes,
        n_epochs=2000,
        scorers={
            'environment': evaluate_on_environment(env),
            'value_scale': average_value_estimation_scorer
        })
