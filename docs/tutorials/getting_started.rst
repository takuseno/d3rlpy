Getting Started
===============

This tutorial is also available on `Google Colaboratory <https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb>`_

Install
-------

First of all, let's install ``d3rlpy`` on your machine::

  $ pip install d3rlpy

See more information at :doc:`/installation`.

.. note::

  If ``core dump`` error occurs in this tutorial, please try
  :ref:`install_from_source`.

.. note::

  ``d3rlpy`` supports Python 3.7+. Make sure which version you use.

.. note::

  If you use GPU, please setup CUDA first.

Prepare Dataset
---------------

You can make your own dataset without any efforts.
In this tutorial, let's use integrated datasets to start.
If you want to make a new dataset, see :doc:`../references/dataset`.

d3rlpy provides suites of datasets for testing algorithms and research.
See more documents at :doc:`../references/datasets`.

.. code-block:: python

  from d3rlpy.datasets import get_cartpole # CartPole-v1 dataset
  from d3rlpy.datasets import get_pendulum # Pendulum-v1 dataset
  from d3rlpy.datasets import get_atari    # Atari 2600 task datasets
  from d3rlpy.datasets import get_d4rl     # D4RL datasets

Here, we use the CartPole dataset to instantly check training results.

.. code-block:: python

  dataset, env = get_cartpole()


Setup Algorithm
---------------

There are many algorithms avaiable in d3rlpy.
Since CartPole is the simple task, let's start from ``DQN``, which is the
Q-learnig algorithm proposed as the first deep reinforcement learning algorithm.

.. code-block:: python

  from d3rlpy.algos import DQNConfig

  # if you don't use GPU, set device=None instead.
  dqn = DQNConfig().create(device="cuda:0")

  # initialize neural networks with the given observation shape and action size.
  # this is not necessary when you directly call fit or fit_online method.
  dqn.build_with_dataset(dataset)

See more algorithms and configurations at :doc:`../references/algos`.

Setup Metrics
-------------

Collecting evaluation metrics is important to train algorithms properly.
d3rlpy provides ``Evaluator`` classes to compute evaluation metrics.

.. code-block:: python

  from d3rlpy.metrics import TDErrorEvaluator

  # calculate metrics with training dataset
  td_error_evaluator = TDErrorEvaluator(episodes=dataset.episodes)

Since evaluating algorithms without access to environment is still difficult,
the algorithm can be directly evaluated with ``EnvironmentEvaluator``
if the environment is available to interact.

.. code-block:: python

  from d3rlpy.metrics import EnvironmentEvaluator

  # set environment in scorer function
  env_evaluator = EnvironmentEvaluator(env)

  # evaluate algorithm on the environment
  rewards = env_evaluator(dqn, dataset=None)

See more metrics and configurations at :doc:`../references/metrics`.


Start Training
--------------

Now, you have everything to start offline training.

.. code-block:: python

  dqn.fit(
      dataset,
      n_steps=10000,
      evaluators={
          'td_error': td_error_evaluator,
          'environment': env_evaluator,
      },
  )

See more about logging at :doc:`../references/logging`.

Once the training is done, your algorithm is ready to make decisions.

.. code-block:: python

  observation = env.reset()

  # return actions based on the greedy-policy
  action = dqn.predict([observation])[0]

  # estimate action-values
  value = dqn.predict_value([observation], [action])[0]

Save and Load
-------------

d3rlpy provides several ways to save trained models.

.. code-block:: python

   import d3rlpy

  # save full parameters and configurations in a single file.
  dqn.save('dqn.d3')
  # load full parameters and build algorithm
  dqn2 = d3rlpy.load_learnable("dqn.d3")

  # save full parameters only
  dqn.save_model('dqn.pt')
  # load full parameters with manual setup
  dqn3 = DQN()
  dqn3.build_with_dataset(dataset)
  dqn3.load_model('dqn.pt')

  # save the greedy-policy as TorchScript
  dqn.save_policy('policy.pt')
  # save the greedy-policy as ONNX
  dqn.save_policy('policy.onnx')

See more information at :doc:`after_training_policies`.
