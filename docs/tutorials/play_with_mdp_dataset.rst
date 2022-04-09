********************
Play with MDPDataset
********************

d3rlpy provides ``MDPDataset``, a dedicated dataset structure for offline RL.
In this tutorial, you can learn how to play with ``MDPDataset``.
Check :doc:`../references/dataset` for more information.

Prepare Dataset
---------------

In this tutorial, let's use a built-in dataset for CartPole-v0.

.. code-block:: python

  # prepare dataset
  dataset, _ = d3rlpy.datasets.get_dataset("cartpole-random")

Understand Episode and Transition
---------------------------------

``MDPDataset`` hierarchically structures the dataset into ``Episode`` and
``Transition``.

.. image:: ../assets/mdp_dataset.png

You can interact with this underlying data structure.

.. code-block:: python

  # first episode
  episode = dataset.episodes[0]

  # access to episode data
  episode.observations
  episode.actions
  episode.rewards

  # first transition
  transition = episode.transitions[0]

  # access to tuple
  transition.observation
  transition.action
  transition.reward
  transition.next_observation

  # linked list structure
  next_transition = transition.next_transition
  assert transition is next_transition.prev_transition

Feed MDPDataset to Algorithm
----------------------------

There are multiple ways to feed datasets to algorithms for offline RL.

.. code-block:: python

  dqn = d3rlpy.algos.DQN()

  # feed as MDPDataset
  dqn.fit(dataset, n_steps=10000)

  # feed as Episode
  dqn.fit(dataset.episodes, n_steps=10000)

  # feed as Transition
  transitions = []
  for episode in dataset.episodes:
      transitions.extend(episode.transitions)
  dqn.fit(transitions, n_steps=10000)

The advantage of this design is that you can split datasets in both
episode-wise and transition-wise.
If you split datasets in episode-wise manner, you can completely remove all
transitions included in test episodes, which makes valiadtion work better.

.. code-block:: python

  # use scikit-learn utility
  from sklearn.model_selection import train_test_split

  # episode-wise split
  train_episodes, test_episodes = train_test_split(dataset.episodes)

  # setup metrics
  metrics = {
    "soft_opc": d3rlpy.metrics.scorer.soft_opc_scorer(return_threshold=180),
    "initial_value": d3rlpy.metrics.scorer.initial_state_value_estimation_scorer,
  }

  # start training with episode-wise splits
  dqn.fit(
      train_episodes,
      n_steps=10000,
      scorers=metrics,
      eval_episodes=test_episodes,
  )

Mix Datasets
------------

You can also mix multiple datasets to train algorithms.

.. code-block:: python

  replay_dataset, _ = d3rlpy.datasets.get_dataset("cartpole-replay")

  # extends replay dataset with random dataset
  replay_dataset.extend(dataset)

  # you can also save it and load it later
  replay_dataset.dump("mixed_dataset.h5")
  mixed_dataset = MDPDataset.load("mixed_dataset.h5")
