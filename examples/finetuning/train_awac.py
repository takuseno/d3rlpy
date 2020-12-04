from d3rlpy.algos import AWAC
from d3rlpy.encoders import VectorEncoderFactory
from d3rlpy.datasets import get_pybullet
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import discounted_sum_of_advantage_scorer
from d3rlpy.metrics.scorer import average_value_estimation_scorer
from sklearn.model_selection import train_test_split

# prepare dataset and environment
dataset, env = get_pybullet('hopper-bullet-random-v0')
_, eval_env = get_pybullet('hopper-bullet-random-v0')

train_episodes, test_episodes = train_test_split(dataset)

encoder_factory = VectorEncoderFactory(hidden_units=[256, 256, 256, 256])

# setup algorithm
awac = AWAC(actor_encoder_factory=encoder_factory,
            critic_encoder_factory=encoder_factory,
            use_gpu=True)

## pretrain
awac.fit(train_episodes[:10000],
         eval_episodes=test_episodes,
         n_epochs=30,
         scorers={
             'environment': evaluate_on_environment(env),
             'advantage': discounted_sum_of_advantage_scorer,
             'value_scale': average_value_estimation_scorer
         })

# fine-tuning
awac.fit_online(env,
                ReplayBuffer(1000000, env, train_episodes[:10000]),
                n_steps=100000,
                eval_env=eval_env,
                eval_epsilon=0.0,
                n_steps_per_epoch=10000)
