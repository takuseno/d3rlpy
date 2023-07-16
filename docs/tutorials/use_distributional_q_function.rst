*****************************
Use Distributional Q-Function
*****************************

The one of the unique features in d3rlpy is to use distributional Q-functions
with arbitrary d3rlpy algorithms.
The distributional Q-functions are powerful and potentially capable of
improving performance of any algorithms.
In this tutorial, you can learn how to use them.
Check :doc:`../references/q_functions` for more information.

.. code-block:: python

  # default standard Q-function
  mean_q_function = d3rlpy.models.MeanQFunctionFactory()
  sac = d3rlpy.algos.SACConfig(q_func_factory=mean_q_function).create()

  # Quantile Regression Q-function
  qr_q_function = d3rlpy.models.QRQFunctionFactory(n_quantiles=200)
  sac = d3rlpy.algos.SACConfig(q_func_factory=qr_q_function).create()

  # Implicit Quantile Network Q-function
  iqn_q_function = d3rlpy.models.IQNQFunctionFactory(
      n_quantiles=32,
      n_greedy_quantiles=64,
      embed_size=64,
  )
  sac = d3rlpy.algos.SACConfig(q_func_factory=iqn_q_function).create()
