from d3rlpy.algos import CQL
from d3rlpy.datasets import get_d4rl
from d3rlpy.models.encoders import VectorEncoderFactory
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

dataset, env = get_d4rl('hopper-medium-v0')

_, test_episodes = train_test_split(dataset, test_size=0.2)

encoder = VectorEncoderFactory(hidden_units=[256, 256, 256])

cql = CQL(actor_encoder_factory=encoder,
          critic_encoder_factory=encoder,
          alpha_learning_rate=0.0,
          use_gpu=True)

cql.fit(dataset.episodes,
        eval_episodes=test_episodes,
        n_epochs=2000,
        scorers={
            'environment': evaluate_on_environment(env),
            'value_scale': average_value_estimation_scorer
        })
