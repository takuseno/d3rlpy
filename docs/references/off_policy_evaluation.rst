Off-Policy Evaluation
=====================

.. module:: d3rlpy.ope

The off-policy evaluation is a method to estimate the trained policy
performance only with offline datasets.

.. code-block:: python

   from d3rlpy.algos import CQL
   from d3rlpy.datasets import get_pybullet

   # prepare the trained algorithm
   cql = CQL.from_json('<path-to-json>/params.json')
   cql.load_model('<path-to-model>/model.pt')

   # dataset to evaluate with
   dataset, env = get_pybullet('hopper-bullet-mixed-v0')

   from d3rlpy.ope import FQE

   # off-policy evaluation algorithm
   fqe = FQE(algo=cql)

   # metrics to evaluate with
   from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
   from d3rlpy.metrics.scorer import soft_opc_scorer

   # train estimators to evaluate the trained policy
   fqe.fit(dataset.episodes,
           eval_episodes=dataset.episodes,
           scorers={
              'init_value': initial_state_value_estimation_scorer,
              'soft_opc': soft_opc_scorer(return_threshold=600)
           })

The evaluation during fitting is evaluating the trained policy.

For continuous control algorithms
---------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.ope.FQE


For discrete control algorithms
-------------------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.ope.DiscreteFQE
