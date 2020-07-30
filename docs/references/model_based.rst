Model-based Data Augmentation (experimental)
============================================

.. module:: d3rlpy.dynamics

d3rlpy provides model-based reinforcement learning algorithms.
In d3rlpy, model-based algorithms are viewed as data augmentation techniques,
which can boost performance potentially beyond the model-free algorithms.

.. code-block:: python

  from d3rlpy.datasets import get_pendulum
  from d3rlpy.dynamics import MOPO
  from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
  from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
  from sklearn.model_selection import train_test_split

  dataset, _ = get_cartpole()

  train_episodes, test_episodes = train_test_split(dataset)

  mopo = MOPO()

  # same as algorithms
  mopo.fit(train_episodes,
           eval_episodes=test_episodes,
           scorers={
              'observation_error': dynamics_observation_prediction_error_scorer,
              'reward_error': dynamics_reward_prediction_error_scorer,
           })


  from d3rlpy.algos import CQL

  # give mopo as dynamics argument.
  cql = CQL(dynamics=mopo)


If you pass a dynamics model to algorithms, new transitions are generated at
the beginning of every epoch.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dynamics.mopo.MOPO
