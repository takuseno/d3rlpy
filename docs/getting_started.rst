Getting Started
===============

This tutorial is also available on `Google Colaboratory <https://colab.research.google.com/github/takuseno/d3rlpy/blob/master/tutorials/cartpole.ipynb>`_

Install
-------

First of all, let's install ``d3rlpy`` on your machine::

  $ pip install d3rlpy

.. note::

  ``d3rlpy`` supports Python 3.6+. Make sure which version you use.

.. note::

  If you use GPU, please setup CUDA first.

Prepare Dataset
---------------

You can make your own dataset without any efforts.
In this tutorial, let's use integrated datasets to start.
If you want to make a new dataset, see :doc:`references/dataset`.

d3rlpy provides suites of datasets for testing algorithms and research.
See more documents at :doc:`references/datasets`.

.. code-block:: python

  from d3rlpy.datasets import get_cartpole # CartPole-v0 dataset
  from d3rlpy.datasets import get_pendulum # Pendulum-v0 dataset
  from d3rlpy.datasets import get_pybullet # PyBullet task datasets
  from d3rlpy.datasets import get_atari    # Atari 2600 task datasets

Here, we use the CartPole dataset to instantly check training results.

.. code-block:: python

  dataset, env = get_cartpole()

One interesting feature of d3rlpy is full compatibility with scikit-learn
utilities.
You can split ``dataset`` into a training dataset and a test dataset just
like supervised learning as follows.

.. code-block:: python

  from sklearn.model_selection import train_test_split

  train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)


Setup Algorithm
---------------

There are many algorithms avaiable in d3rlpy.
Since CartPole is the simple task, let's start from ``DQN``, which is the
Q-learnig algorithm proposed as the first deep reinforcement learning algorithm.

.. code-block:: python

  from d3rlpy.algos import DQN

  # if you don't use GPU, set use_gpu=False instead.
  dqn = DQN(use_gpu=True)

  # initialize neural networks with the given observation shape and action size.
  # this is not necessary when you directly call fit or fit_online method.
  dqn.build_with_dataset(dataset)

See more algorithms and configurations at :doc:`references/algos`.

Setup Metrics
-------------

Collecting evaluation metrics is important to train algorithms properly.
In d3rlpy, the metrics is computed through scikit-learn style scorer functions.

.. code-block:: python

  from d3rlpy.metrics.scorer import td_error_scorer
  from d3rlpy.metrics.scorer import average_value_estimation_scorer

  # calculate metrics with test dataset
  td_error = td_error_scorer(dqn, test_episodes)

Since evaluating algorithms without access to environment is still difficult,
the algorithm can be directly evaluated with ``evaluate_on_environment`` function
if the environment is available to interact.

.. code-block:: python

  from d3rlpy.metrics.scorer import evaluate_on_environment

  # set environment in scorer function
  evaluate_scorer = evaluate_on_environment(env)

  # evaluate algorithm on the environment
  rewards = evaluate_scorer(dqn)

See more metrics and configurations at :doc:`references/metrics`.


Start Training
--------------

Now, you have all to start data-driven training.

.. code-block:: python

  dqn.fit(train_episodes,
          eval_episodes=test_episodes,
          n_epochs=10,
          scorers={
              'td_error': td_error_scorer,
              'value_scale': average_value_estimation_scorer,
              'environment': evaluate_scorer
          })

Then, you will see training progress in the console like below::

  augmentation=[]
  batch_size=32
  bootstrap=False
  dynamics=None
  encoder_params={}
  eps=0.00015
  gamma=0.99
  learning_rate=6.25e-05
  n_augmentations=1
  n_critics=1
  n_frames=1
  q_func_type=mean
  scaler=None
  share_encoder=False
  target_update_interval=8000.0
  use_batch_norm=True
  use_gpu=None
  observation_shape=(4,)
  action_size=2
  100%|███████████████████████████████████| 2490/2490 [00:24<00:00, 100.63it/s]
  epoch=0 step=2490 value_loss=0.190237
  epoch=0 step=2490 td_error=1.483964
  epoch=0 step=2490 value_scale=1.241220
  epoch=0 step=2490 environment=157.400000
  100%|███████████████████████████████████| 2490/2490 [00:24<00:00, 100.63it/s]
  .
  .
  .

See more about logging at :doc:`references/logging`.

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

  # save full parameters
  dqn.save_model('dqn.pt')

  # load full parameters
  dqn2 = DQN()
  dqn2.build_with_dataset(dataset)
  dqn2.load_model('dqn.pt')

  # save the greedy-policy as TorchScript
  dqn.save_policy('policy.pt')

  # save the greedy-policy as ONNX
  dqn.save_policy('policy.onnx', as_onnx=True)

See more information at :doc:`/references/save_and_load`.
