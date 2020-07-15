from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import td_error_scorer
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

# obtain dataset
dataset, env = get_cartpole()

# split traeining dataset and test dataset
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# setup algorithm
dqn = DQN(n_epochs=1)

# train
dqn.fit(train_episodes,
        eval_episodes=test_episodes,
        scorers={
            'environment': evaluate_on_environment(env),
            'td_error': td_error_scorer,
            'discounted_advantage': discounted_sum_of_advantage_scorer,
            'value_scale': average_value_estimation_scorer
        })

# evaluate trained algorithm
evaluate_on_environment(env, render=True)(dqn)
