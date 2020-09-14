from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL
from d3rlpy.metrics.ope import FQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from sklearn.model_selection import train_test_split

dataset, env = get_pybullet('hopper-bullet-mixed-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# train algorithm
cql = CQL(n_epochs=100, use_gpu=True)
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        scorers={
            'environment': evaluate_on_environment(env),
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(600)
        })

# or load the trained model
# cql = CQL.from_json('<path-to-json>/params.json')
# cql.load_model('<path-to-model>/model.pt')

# evaluate the trained policy
fqe = FQE(algo=cql,
          n_epochs=200,
          q_func_type='qr',
          learning_rate=1e-4,
          use_gpu=True,
          encoder_params={'hidden_units': [1024, 1024, 1024, 1024]})
fqe.fit(dataset.episodes,
        eval_episodes=dataset.episodes,
        scorers={
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(600)
        })
