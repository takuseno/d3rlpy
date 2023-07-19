************************
Preprocess / Postprocess
************************

In this tutorial, you can learn how to preprocess datasets and postprocess
continuous action outputs.
Please check :doc:`../references/preprocessing` for more information.

Preprocess Observations
-----------------------

If your dataset includes unnormalized observations, you can normalize or
standardize the observations by specifying ``observation_scaler`` argument.
In this case, the statistics of the dataset will be computed at the beginning
of offline training.

.. code-block:: python

  import d3rlpy

  dataset, _ = d3rlpy.datasets.get_dataset("pendulum-random")

  # prepare scaler without initialization
  observation_scaler = d3rlpy.preprocessing.StandardObservationScaler()

  sac = d3rlpy.algos.SACConfig(observation_scaler=observation_scaler).create()

Alternatively, you can manually instantiate preprocessing parameters.

.. code-block:: python

  # setup manually
  observations = []
  for episode in dataset.episodes:
      observations += episode.observations.tolist()
  mean = np.mean(observations, axis=0)
  std = np.std(observations, axis=0)
  observation_scaler = d3rlpy.preprocessing.StandardObservationScaler(mean=mean, std=std)

  # set as observation_scaler
  sac = d3rlpy.algos.SACConfig(observation_scaler=observation_scaler).create()


Please check :doc:`../references/preprocessing` for the full list of available
observation preprocessors.

Preprocess / Postprocess Actions
--------------------------------

In training with continuous action-space, the actions must be in the range
between ``[-1.0, 1.0]`` due to the underlying ``tanh`` activation at the policy
functions.
In d3rlpy, you can easily normalize inputs and denormalize outpus instead of
normalizing datasets by yourself.

.. code-block:: python

  # prepare scaler without initialization
  action_scaler = d3rlpy.preprocessing.MinMaxActionScaler()

  # set as action scaler
  sac = d3rlpy.algos.SACConfig(action_scaler=action_scaler).create()

  # setup manually
  actions = []
  for episode in dataset.episodes:
      actions += episode.actions.tolist()
  minimum_action = np.min(actions, axis=0)
  maximum_action = np.max(actions, axis=0)
  action_scaler = d3rlpy.preprocessing.MinMaxActionScaler(
      minimum=minimum_action,
      maximum=maximum_action,
  )

  # set as action scaler
  sac = d3rlpy.algos.SACConfig(action_scaler=action_scaler).create()

Please check :doc:`../references/preprocessing` for the full list of available
action preprocessors.


Preprocess Rewards
------------------

The effect of scaling rewards is not well studied yet in RL community, however,
it's confirmed that the reward scale affects training performance.

.. code-block:: python

  # prepare scaler without initialization
  reward_scaler = d3rlpy.preprocessing.StandardRewardScaler()

  # set as reward scaler
  sac = d3rlpy.algos.SACConfig(reward_scaler=reward_scaler).create()

  # setup manuall
  rewards = []
  for episode in dataset.episodes:
      rewards += episode.rewards.tolist()
  mean = np.mean(rewards)
  std = np.std(rewards)
  reward_scaler = StandardRewardScaler(mean=mean, std=std)

  # set as reward scaler
  sac = d3rlpy.algos.SACConfig(reward_scaler=reward_scaler).create()


Please check :doc:`../references/preprocessing` for the full list of available
reward preprocessors.
