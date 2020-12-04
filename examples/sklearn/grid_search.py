from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.context import parallel
from sklearn.model_selection import GridSearchCV

# obtain dataset
dataset, env = get_cartpole()

# setup algowithm with GPU enabled
dqn = DQN(use_gpu=True)

# grid search with multiple GPUs assigned to individual processs
with parallel():
    env_score = evaluate_on_environment(env)
    gscv = GridSearchCV(estimator=dqn,
                        param_grid={
                            'learning_rate': [1e-3, 3e-4, 1e-4],
                            'gamma': [0.99, 0.95, 0.9]
                        },
                        scoring={'environment': env_score},
                        refit=False,
                        n_jobs=3)
    gscv.fit(dataset.episodes, n_epochs=1, show_progress=False)

print(gscv.grid_scores_)
