Off-Policy Evaluation
=====================

.. module:: d3rlpy.ope

The off-policy evaluation is a method to estimate the trained policy
performance only with offline datasets.

.. code-block:: python

   import d3rlpy

   # prepare the trained algorithm
   cql = d3rlpy.load_learnable("model.d3")

   # dataset to evaluate with
   dataset, env = d3rlpy.datasets.get_pendulum()

   # off-policy evaluation algorithm
   fqe = d3rlpy.ope.FQE(algo=cql, config=d3rlpy.ope.FQEConfig())

   # train estimators to evaluate the trained policy
   fqe.fit(
      dataset,
      n_steps=100000,
      scorers={
         'init_value': d3rlpy.metrics.InitialStateValueEstimationEvaluator(),
         'soft_opc': d3rlpy.metrics.SoftOPCEvaluator(return_threshold=-300),
      },
   )

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
