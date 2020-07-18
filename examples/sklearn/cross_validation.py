from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.context import parallel
from sklearn.model_selection import cross_validate

# obtain dataset
dataset, env = get_cartpole()

# setup algowithm with GPU enabled
dqn = DQN(n_epochs=1, use_gpu=True)

# cross validation with multiple GPUs assigned to individual processs
with parallel():
    env_score = evaluate_on_environment(env)
    scores = cross_validate(dqn,
                            dataset,
                            fit_params={'show_progress': False},
                            scoring={'environment': env_score},
                            n_jobs=3)  # 3 parallel training processes

print(scores)
