*****************************
Use Distributional Q-Function
*****************************

The one of the unique features in d3rlpy is to use distributional Q-functions
with arbitrary d3rlpy algorithms.
The distributional Q-functions are powerful and potentially capable of
improving performance of any algorithms.
In this tutorial, you can learn how to use them.
Check :doc:`../references/q_functions` for more information.

Specify by String Alias
-----------------------

The supported Q-functions can be specified by string alias.
In this case, the default hyper-parameters will be used for the Q-function.

.. code-block:: python

  import d3rlpy

  # default standard Q-function
  sac = d3rlpy.algos.SAC(q_func_factory="mean")

  # Quantile Regression Q-function
  sac = d3rlpy.algos.SAC(q_func_factory="qr")

  # Implicit Quantile Network Q-function
  sac = d3rlpy.algos.SAC(q_func_factory="iqn")

Specify by instantiating QFunctionFactory
-----------------------------------------

If you want to specify hyper-parameters, you need to instantiate a
``QFunctionFactory`` object.

.. code-block:: python

  # default standard Q-function
  mean_q_function = d3rlpy.models.q_functions.MeanQFunctionFactory()
  sac = d3rlpy.algos.SAC(q_func_factory=mean_q_function)

  # Quantile Regression Q-function
  qr_q_function = d3rlpy.models.q_functions.QRQFunctionFactory(n_quantiles=200)
  sac = d3rlpy.algos.SAC(q_func_factory=qr_q_function)

  # Implicit Quantile Network Q-function
  iqn_q_function = d3rlpy.models.q_functions.IQNQFunctionFactory(
      n_quantiles=32,
      n_greedy_quantiles=64,
      embed_size=64,
  )
  sac = d3rlpy.algos.SAC(q_func_factory=iqn_q_function)
