***********************
Create Your Dataset
***********************

The data collection API is introduced in :doc:`data_collection`.
In this tutorial, you can learn how to build your dataset from logged data
such as the user data collected in your web service.

Prepare Logged Data
-------------------

First of all, you need to prepare your logged data.
In this tutorial, let's use randomly generated data.
``terminals`` represents the last step of episodes.
If ``terminals[i] == 1.0``, i-th step is the terminal state.
Otherwise you need to set zeros for non-terminal states.

.. code-block:: python

  import numpy as np

  # vector observation
  # 1000 steps of observations with shape of (100,)
  observations = np.random.random((1000, 100))

  # 1000 steps of actions with shape of (4,)
  actions = np.random.random((1000, 4))

  # 1000 steps of rewards
  rewards = np.random.random(1000)

  # 1000 steps of terminal flags
  terminals = np.random.randint(2, size=1000)

Build MDPDataset
----------------

Once your logged data is ready, you can build ``MDPDataset`` object.

.. code-block:: python

  import d3rlpy

  dataset = d3rlpy.dataset.MDPDataset(
      observations=observations,
      actions=actions,
      rewards=rewards,
      terminals=terminals,
  )

Set Timeout Flags
-----------------

In RL, there is the case where you want to stop an episode without a terminal
state.
For example, if you're collecting data of a 4-legged robot walking forward,
the walking task basically never ends as long as the robot keeps walking while
the logged episode must stop somewhere.
In this case, you can use ``episode_terminals`` to represent this timeout states.

.. code-block:: python

  # terminal states
  terminals = np.zeros(1000)

  # timeout states
  episode_terminals = np.random.randint(2, size=1000)

  dataset = d3rlpy.dataset.MDPDataset(
      observations=observations,
      actions=actions,
      rewards=rewards,
      terminals=terminals,
      episode_terminals=episode_terminals,
  )
