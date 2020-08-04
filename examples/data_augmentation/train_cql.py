from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split


dataset, env = get_atari('breakout-expert-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

cql = DiscreteCQL(n_epochs=100,
                  scaler='pixel',
                  augmentation=['random_shift', 'intensity'],
                  use_batch_norm=False,
                  use_gpu=True)

cql.fit(train_episodes,
        eval_episodes=test_episodes,
        scorers={
            'environment': evaluate_on_environment(env, epsilon=0.05),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        })
