from sklearn.model_selection import train_test_split
from d3rlpy.datasets import get_atari
from d3rlpy.algos import DiscreteCQL
from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer

dataset, env = get_atari('breakout-expert-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# train algorithm
cql = DiscreteCQL(n_epochs=100,
                  scaler='pixel',
                  q_func_factory='qr',
                  n_frames=4,
                  use_gpu=True)
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        scorers={
            'environment': evaluate_on_environment(env, epsilon=0.05),
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(70)
        })

# or load the trained model
# cql = DiscreteCQL.from_json('<path-to-json>/params.json')
# cql.load_model('<path-to-model>/model.pt')

# evaluate the trained policy
fqe = DiscreteFQE(algo=cql,
                  n_epochs=100,
                  q_func_factory='qr',
                  learning_rate=1e-4,
                  scaler='pixel',
                  n_frames=4,
                  discrete_action=True,
                  use_gpu=True)
fqe.fit(dataset.episodes,
        eval_episodes=dataset.episodes,
        scorers={
            'init_value': initial_state_value_estimation_scorer,
            'soft_opc': soft_opc_scorer(70)
        })
