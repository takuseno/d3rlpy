from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment

# obtain dataset
dataset, env = get_cartpole()

# setup algorithm
dqn = DQN()

# train
dqn.fit(dataset.episodes, n_epochs=1)

# evaluate trained algorithm
evaluate_on_environment(env, render=True)(dqn)
