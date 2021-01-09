Q Functions
===========

.. module:: d3rlpy.q_functions

d3rlpy provides various Q functions including state-of-the-arts, which are
internally used in algorithm objects.
You can switch Q functions by passing ``q_func_factory`` argument at
algorithm initialization.

.. code-block:: python

  from d3rlpy.algos import CQL

  cql = CQL(q_func_factory='qr') # use Quantile Regression Q function

Also you can change hyper parameters.

.. code-block:: python

   from d3rlpy.models.q_functions import QRQFunctionFactory

   q_func = QRQFunctionFactory(n_quantiles=32)

   cql = CQL(q_func_factory=q_func)

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

   d3rlpy.models.q_functions.MeanQFunctionFactory
   d3rlpy.models.q_functions.QRQFunctionFactory
   d3rlpy.models.q_functions.IQNQFunctionFactory
   d3rlpy.models.q_functions.FQFQFunctionFactory
