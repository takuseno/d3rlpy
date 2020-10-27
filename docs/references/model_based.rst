Model-based Data Augmentation
=============================

.. module:: d3rlpy.dynamics

d3rlpy provides model-based reinforcement learning algorithms.
In d3rlpy, model-based algorithms are viewed as data augmentation techniques,
which can boost performance potentially beyond the model-free algorithms.

.. code-block:: python

  from d3rlpy.datasets import get_pendulum
  from d3rlpy.dynamics import MOPO
  from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
  from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
  from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer
  from sklearn.model_selection import train_test_split

  dataset, _ = get_pendulum()

  train_episodes, test_episodes = train_test_split(dataset)

  mopo = MOPO(learning_rate=1e-4, use_gpu=True)

  # same as algorithms
  mopo.fit(train_episodes,
           eval_episodes=test_episodes,
           n_epochs=100,
           scorers={
              'observation_error': dynamics_observation_prediction_error_scorer,
              'reward_error': dynamics_reward_prediction_error_scorer,
              'variance': dynamics_prediction_variance_scorer,
           })

Pick the best model based on evaluation metrics.

.. code-block:: python

  from d3rlpy.dynamics import MOPO
  from d3rlpy.algos import CQL

  # load trained dynamics model
  mopo = MOPO.from_json('<path-to-params.json>/params.json')
  mopo.load_model('<path-to-model>/model_xx.pt')
  mopo.n_transitions = 400 # tunable parameter
  mopo.horizon = 5 # tunable parameter
  mopo.lam = 1.0 # tunable parameter

  # give mopo as dynamics argument.
  cql = CQL(dynamics=mopo)

If you pass a dynamics model to algorithms, new transitions are generated at
the beginning of every epoch.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dynamics.mopo.MOPO
