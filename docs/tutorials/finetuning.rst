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
  dqn = d3rlpy.algos.DQN()

  # start offline training
  dqn.fit(dataset, n_steps=100000)

Finetune with Environment
-------------------------

.. code-block:: python

  # setup experience replay buffer
  buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

  # setup exploration strategy if necessary
  explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.1)

  # start finetuning
  dqn.fit_online(env, buffer, explorer, n_steps=100000)

Finetune with Saved Policy
--------------------------

If you want to finetune the saved policy, that's also easy to do with d3rlpy.

.. code-block:: python

  # setup algorithm
  dqn = d3rlpy.algos.DQN()

  # initialize neural networks before loading parameters
  dqn.build_with_env(env)

  # load pretrained policy
  dqn.load_model("dqn_model.pt")

  # start finetuning
  dqn.fit_online(env, buffer, explorer, n_steps=100000)
