Tips
====

Reproducibility
---------------

Reproducibility is one of the most important things when doing research
activity.
Here is a simple example in d3rlpy.

.. code-block:: python

  import d3rlpy
  import gym

  # set random seeds in random module, numpy module and PyTorch module.
  d3rlpy.seed(313)

  # set environment seed
  env = gym.make('Hopper-v2')
  d3rlpy.envs.seed_env(env, 313)

Learning from image observation
-------------------------------

d3rlpy supports both vector observations and image observations.
There are several things you need to care about if you want to train RL agents from
image observations.

.. code-block:: python

  import d3rlpy

  # observation MUST be uint8 array, and the channel-first images
  observations = np.random.randint(256, size=(100000, 1, 84, 84), dtype=np.uint8)
  actions = np.random.randomint(4, size=100000)
  rewards = np.random.random(100000)
  terminals = np.random.randint(2, size=100000)

  dataset = d3rlpy.dataset.MDPDataset(
      observations=observations,
      actions=actions,
      rewards=rewards,
      terminals=terminals,
      # stack last 4 frames (stacked shape is [4, 84, 84])
      transition_picker=d3rlpy.dataset.FrameStackTransitionPicker(n_frames=4),
  )

  dqn = DQNConfig(
      observation_scaler=d3rlpy.preprocessing.PixelObservationScaler(),  # pixels are devided by 255
  ).create()

Improve performance beyond the original paper
---------------------------------------------

d3rlpy provides many options that you can use to improve performance potentially
beyond the original paper.
All the options are powerful, but the best combinations and hyperparameters are
always dependent on the tasks.

.. code-block:: python

  import d3rlpy

  # use batch normalization
  # this seems to improve performance with discrete action-spaces
  encoder = d3rlpy.models.DefaultEncoderFactory(use_batch_norm=True)
  # use distributional Q function leading to robust improvement
  q_func = d3rlpy.models.QRQFunctionFactory()
  dqn = d3rlpy.algos.DQNConfig(
      encoder_factory=encoder,
      q_func_factory=q_func,
  ).create()

  # use dropout
  # this could dramatically improve performance
  encoder = d3rlpy.models.DefaultEncoderFactory(dropout_rate=0.2)
  sac = d3rlpy.algos.SACConfig(actor_encoder_factory=encoder).create()

  # multi-step transition sampling
  transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
      n_steps=3,
      gamma=0.99,
  )
  # replay buffer for experience replay
  buffer = d3rlpy.dataset.create_fifo_replay_buffer(
      limit=100000,
      env=env,
      transition_picker=transition_picker,
  )
