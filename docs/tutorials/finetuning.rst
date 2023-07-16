**********
Finetuning
**********

d3rlpy supports smooth transition from offline training to online training.

Prepare Dataset and Environment
-------------------------------

In this tutorial, let's use a built-in dataset for CartPole-v0 environment.

.. code-block:: python

  import d3rlpy

  # setup random CartPole-v0 dataset and environment
  dataset, env = d3rlpy.datasets.get_dataset("cartpole-random")

Pretrain with Dataset
---------------------

.. code-block:: python

  # setup algorithm
  dqn = d3rlpy.algos.DQNConfig().create()

  # start offline training
  dqn.fit(dataset, n_steps=100000)

Finetune with Environment
-------------------------

.. code-block:: python

  # setup experience replay buffer
  buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

  # setup exploration strategy if necessary
  explorer = d3rlpy.algos.ConstantEpsilonGreedy(0.1)

  # start finetuning
  dqn.fit_online(env, buffer, explorer, n_steps=100000)

Finetune with Saved Policy
--------------------------

If you want to finetune the saved policy, that's also easy to do with d3rlpy.

.. code-block:: python

  # setup algorithm
  dqn = d3rlpy.load_learnable("dqn_model.d3")

  # start finetuning
  dqn.fit_online(env, buffer, explorer, n_steps=100000)

Finetune with Different Algorithm
---------------------------------

If you want to finetune the saved policy trained offline with online RL
algorithms, you can do it in an out-of-the-box way.

.. code-block:: python

  # setup offline RL algorithm
  cql = d3rlpy.algos.DiscreteCQLConfig().create()

  # train offline
  cql.fit(dataset, n_steps=100000)

  # transfer to DQN
  dqn = d3rlpy.algos.DQNConfig().create()
  dqn.copy_q_function_from(cql)

  # start finetuning
  dqn.fit_online(env, buffer, explorer, n_steps=100000)

In actor-critic cases, you should also transfer the policy function.

.. code-block:: python

  # offline RL
  cql = d3rlpy.algos.CQLConfig().create()
  cql.fit(dataset, n_steps=100000)

  # transfer to SAC
  sac = d3rlpy.algos.SACConfig().create()
  sac.build_with_env(env)
  sac.copy_q_function_from(cql)
  sac.copy_policy_from(cql)

  # online RL
  sac.fit_online(env, buffer, n_steps=100000)
