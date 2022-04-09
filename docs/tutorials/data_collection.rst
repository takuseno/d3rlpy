***************
Data Collection
***************

d3rlpy provides APIs to support data collection from environments.
This feature is specifically useful if you want to build your own original
datasets for research or practice purposes.

Prepare Environment
-------------------

d3rlpy supports environments with OpenAI Gym interface.
In this tutorial, let's use simple CartPole environment.

.. code-block:: python

  import gym

  env = gym.make("CartPole-v0")

Data Collection with Random Policy
----------------------------------

If you want to collect experiences with uniformly random policy, you can use
``RandomPolicy`` and ``DiscreteRandomPolicy``.
This procedure corresponds to ``random`` datasets in D4RL.

.. code-block:: python

  import d3rlpy

  # setup algorithm
  random_policy = d3rlpy.algos.DiscreteRandomPolicy()

  # prepare experience replay buffer
  buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

  # start data collection
  random_policy.collect(env, buffer, n_steps=100000)

  # export as MDPDataset
  dataset = buffer.to_mdp_dataset()

  # save MDPDataset
  dataset.dump("random_policy_dataset.h5")

Data Collection with Trained Policy
-----------------------------------

If you want to collect experiences with previously trained policy, you can
still use the same set of APIs.
This procedure corresponds to ``medium`` datasets in D4RL.

.. code-block:: python

  # setup algorithm
  dqn = d3rlpy.algos.DQN()

  # initialize neural networks before loading parameters
  dqn.build_with_env(env)

  # load pretrained parameters
  dqn.load_model("dqn_model.pt")

  # prepare experience replay buffer
  buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

  # start data collection
  dqn.collect(env, buffer, n_steps=100000)

  # export as MDPDataset
  dataset = buffer.to_mdp_dataset()

  # save MDPDataset
  dataset.dump("trained_policy_dataset.h5")

Data Collection while Training Policy
-------------------------------------

If you want to use experiences collected during training to build a new dataset,
you can simply use ``fit_online`` and save the dataset.
This procedure corresponds to ``replay`` datasets in D4RL.

.. code-block:: python

  # setup algorithm
  dqn = d3rlpy.algos.DQN()

  # prepare experience replay buffer
  buffer = d3rlpy.online.buffers.ReplayBuffer(maxlen=100000, env=env)

  # prepare exploration strategy if necessary
  explorer = d3rlpy.online.explorers.ConstantEpsilonGreedy(0.3)

  # start data collection
  dqn.fit_online(env, buffer, n_steps=100000)

  # export as MDPDataset
  dataset = buffer.to_mdp_dataset()

  # save MDPDataset
  dataset.dump("replay_dataset.h5")
