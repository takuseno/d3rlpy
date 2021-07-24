(experimental) Model-based Algorithms
=====================================

.. module:: d3rlpy.dynamics

d3rlpy provides model-based reinforcement learning algorithms.

.. code-block:: python

  from d3rlpy.datasets import get_pendulum
  from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
  from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
  from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
  from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer
  from sklearn.model_selection import train_test_split

  dataset, _ = get_pendulum()

  train_episodes, test_episodes = train_test_split(dataset)

  dynamics = d3rlpy.dynamics.ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=True)

  # same as algorithms
  dynamics.fit(train_episodes,
               eval_episodes=test_episodes,
               n_epochs=100,
               scorers={
                  'observation_error': dynamics_observation_prediction_error_scorer,
                  'reward_error': dynamics_reward_prediction_error_scorer,
                  'variance': dynamics_prediction_variance_scorer,
               })

Pick the best model and pass it to the model-based RL algorithm.

.. code-block:: python

  from d3rlpy.algos import MOPO

  # load trained dynamics model
  dynamics = ProbabilisticEnsembleDynamics.from_json('<path-to-params.json>/params.json')
  dynamics.load_model('<path-to-model>/model_xx.pt')

  # give mopo as generator argument.
  mopo = MOPO(dynamics=dynamics)

Dynamics Model
--------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dynamics.ProbabilisticEnsembleDynamics
