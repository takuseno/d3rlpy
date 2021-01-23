from d3rlpy.algos import BEAR
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from d3rlpy.models.encoders import VectorEncoderFactory
from sklearn.model_selection import train_test_split

dataset, env = get_d4rl('hopper-medium-v0')

_, test_episodes = train_test_split(dataset, test_size=0.2)

vae_encoder = VectorEncoderFactory(hidden_units=[750, 750])

bear = BEAR(imitator_encoder_factory=vae_encoder,
            mmd_kernel='gaussian',
            use_gpu=True)

bear.fit(dataset.episodes,
         eval_episodes=dataset.episodes,
         n_epochs=2000,
         scorers={
             'environment': evaluate_on_environment(env),
             'value_scale': average_value_estimation_scorer
         })
