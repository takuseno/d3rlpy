Q Functions
===========

.. module:: d3rlpy.models.torch.q_functions

d3rlpy provides various Q functions including state-of-the-arts, which are
internally used in algorithm objects.
You can switch Q functions by passing `q_func_type` argument at
algorithm initialization.

.. code-block:: python

  from d3rlpy.algos import CQL

  cql = CQL(q_func_type='qr') # use Quantile Regression Q function

The default Q function is `mean` approximator, which estimates expected scalar
action-values.
However, in recent advancements of deep reinforcement learning, the new type
of action-value approximators has been proposed, which is called
`distributional` Q functions.

Unlike the `mean` approximator, the `distributional` Q functions estimate
distribution of action-values.
This `distributional` approaches have shown consistently much stronger
performance than the `mean` approximator.

Here is a list of available Q functions in the order of performance
ascendingly.
Currently, as a trade-off between performance and computational complexity,
the higher performance requires the more expensive computational costs.

.. list-table:: available Q functions
   :header-rows: 1

   * - q_func_type
     - reference
   * - mean (default)
     - N/A
   * - qr
     - `Quantile Regression <https://arxiv.org/abs/1710.10044>`_
   * - iqn
     - `Implicit Quantile Network <https://arxiv.org/abs/1806.06923>`_
   * - fqf
     - `Fully-parametrized Quantile Function <https://arxiv.org/abs/1911.02140>`_
