from d3rlpy.algos import CQL
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics.scorer import evaluate_on_environment

dataset, env = get_d4rl('hopper-medium-replay-v0')

cql = CQL(batch_size=256, use_gpu=True)

cql.fit(dataset.episodes,
        eval_episodes=dataset.episodes,
        n_epochs=2000,
        scorers={'environment': evaluate_on_environment(env)})
