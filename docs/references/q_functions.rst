Q Functions
===========

.. module:: d3rlpy.models

d3rlpy provides various Q functions including state-of-the-arts, which are
internally used in algorithm objects.
You can switch Q functions by passing ``q_func_factory`` argument at
algorithm initialization.

.. code-block:: python

  import d3rlpy

  cql = d3rlpy.algos.CQLConfig(q_func_factory=d3rlpy.models.QRQFunctionFactory())

Also you can change hyper parameters.

.. code-block:: python

   q_func = d3rlpy.models.QRQFunctionFactory(n_quantiles=32)

   cql = d3rlpy.algos.CQLConfig(q_func_factory=q_func).create()

The default Q function is ``mean`` approximator, which estimates expected scalar
action-values.
However, in recent advancements of deep reinforcement learning, the new type
of action-value approximators has been proposed, which is called
`distributional` Q functions.

Unlike the ``mean`` approximator, the `distributional` Q functions estimate
distribution of action-values.
This `distributional` approaches have shown consistently much stronger
performance than the ``mean`` approximator.

Here is a list of available Q functions in the order of performance
ascendingly.
Currently, as a trade-off between performance and computational complexity,
the higher performance requires the more expensive computational costs.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.models.MeanQFunctionFactory
   d3rlpy.models.QRQFunctionFactory
   d3rlpy.models.IQNQFunctionFactory
