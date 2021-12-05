from d3rlpy.datasets import get_cartpole
from d3rlpy.algos import DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from sklearn.model_selection import train_test_split

import d3rlpy

# prepare dataset
dataset, env = d3rlpy.datasets.get_pybullet('hopper-bullet-mixed-v0')

# prepare algorithm
cql = d3rlpy.algos.CQL(use_gpu=True)

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=100,
        scorers={
            'environment': d3rlpy.metrics.evaluate_on_environment(env),
            'td_error': d3rlpy.metrics.td_error_scorer,
            'trueQ': true_q_scorer})

fqe = FQE(algo = cql,
        use_gpu = True)

fqe.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs = 5,
        scorers={
            'estimated_q_values': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(500),
            'trueQ': true_q_scorer},
        with_timestamp=False,
        experiment_name = 'DiscreteFQE_v0')
