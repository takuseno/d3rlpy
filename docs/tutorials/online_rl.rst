*********
Online RL
*********

Prepare Environment
-------------------

d3rlpy supports environments with OpenAI Gym interface.
In this tutorial, let's use simple CartPole environment.

.. code-block:: python

  import gym

  # for training
  env = gym.make("CartPole-v1")

  # for evaluation
  eval_env = gym.make("CartPole-v1")

Setup Algorithm
---------------

Just like offline RL training, you can setup an algorithm object.

.. code-block:: python

  import d3rlpy

  # if you don't use GPU, set use_gpu=False instead.
  dqn = d3rlpy.algos.DQNConfig(
      batch_size=32,
      learning_rate=2.5e-4,
      target_update_interval=100,
  ).create(device="cuda:0")

  # initialize neural networks with the given environment object.
  # this is not necessary when you directly call fit or fit_online method.
  dqn.build_with_env(env)


Setup Online RL Utilities
-------------------------

Unlike offline RL training, you'll need to setup an experience replay buffer and
an exploration strategy.

.. code-block:: python

  # experience replay buffer
  buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

  # exploration strategy
  # in this tutorial, epsilon-greedy policy with static epsilon=0.3
  explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.3)


Start Training
--------------

Now, you have everything you need to start online RL training.
Let's put them together!

.. code-block:: python

  dqn.fit_online(
      env,
      buffer,
      explorer,
      n_steps=100000,  # train for 100K steps
      eval_env=eval_env,
      n_steps_per_epoch=1000,  # evaluation is performed every 1K steps
      update_start_step=1000,  # parameter update starts after 1K steps
  )

Train with Stochastic Policy
----------------------------

If the algorithm uses a stochastic policy (e.g. SAC), you can train algorithms
without setting an exploration strategy.

.. code-block:: python

  sac = d3rlpy.algos.DiscreteSACConfig().create()
  sac.fit_online(
      env,
      buffer,
      n_steps=100000,
      eval_env=eval_env,
      n_steps_per_epoch=1000,
      update_start_step=1000,
  )
