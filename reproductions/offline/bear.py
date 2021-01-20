from d3rlpy.algos import BEAR
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics.scorer import evaluate_on_environment

dataset, env = get_d4rl('hopper-medium-replay-v0')

bear = BEAR(use_gpu=True)

bear.fit(dataset.episodes,
         eval_episodes=dataset.episodes,
         n_epochs=2000,
         scorers={'environment': evaluate_on_environment(env)})
