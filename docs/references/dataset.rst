.. _mdp_dataset:

Replay Buffer
=============

.. module:: d3rlpy.dataset

You can also check advanced use cases at `examples <https://github.com/takuseno/d3rlpy/tree/master/examples>`_ directory.

MDPDataset
~~~~~~~~~~

d3rlpy provides useful dataset structure for data-driven deep reinforcement
learning.
In supervised learning, the training script iterates input data :math:`X` and
label data :math:`Y`.
However, in reinforcement learning, mini-batches consist with sets of
:math:`(s_t, a_t, r_t, s_{t+1})` and episode terminal flags.
Converting a set of observations, actions, rewards and terminal flags into this
tuples is boring and requires some codings.

Therefore, d3rlpy provides ``MDPDataset`` class which enables you to handle
reinforcement learning datasets without any efforts.

.. code-block:: python

    import d3rlpy

    # 1000 steps of observations with shape of (100,)
    observations = np.random.random((1000, 100))
    # 1000 steps of actions with shape of (4,)
    actions = np.random.random((1000, 4))
    # 1000 steps of rewards
    rewards = np.random.random(1000)
    # 1000 steps of terminal flags
    terminals = np.random.randint(2, size=1000)

    dataset = d3rlpy.dataset.MDPDataset(observations, actions, rewards, terminals)

    # save as HDF5
    with open("dataset.h5", "wb") as f:
        dataset.dump(f)

    # load from HDF5
    with open("dataset.h5", "wb") as f:
        new_dataset = d3rlpy.dataset.ReplayBuffer.load(f, d3rlpy.dataset.InfiniteBuffer())


Note that the ``observations``, ``actions``, ``rewards`` and ``terminals``
must be aligned with the same timestep.

.. code-block:: python

  observations = [s1, s2, s3, ...]
  actions      = [a1, a2, a3, ...]
  rewards      = [r1, r2, r3, ...]  # r1 = r(s1, a1)
  terminals    = [t1, t2, t3, ...]  # t1 = t(s1, a1)

``MDPDataset`` is actually a shortcut of ``ReplayBuffer`` class.


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.MDPDataset


Replay Buffer
~~~~~~~~~~~~~

``ReplayBuffer`` is a class that represents an experience replay buffer in d3rlpy.
In d3rlpy, ``ReplayBuffer`` is a highly moduralized interface for flexibility.
You can compose sub-components of ``ReplayBuffer``, ``Buffer``, ``TransitionPicker``, `TrajectorySlicer` and ``WriterPreprocess`` to customize experiments.

.. code-block:: python

   import d3rlpy

   # Buffer component
   buffer = d3rlpy.dataset.FIFOBuffer(limit=100000)

   # TransitionPicker component
   transition_picker = d3rlpy.dataset.BasicTransitionPicker()

   # TrajectorySlicer component
   trajectory_slicer = d3rlpy.dataset.BasicTrajectorySlicer()

   # WriterPreprocess component
   writer_preprocessor = d3rlpy.dataset.BasicWriterPreprocess()

   # Need to specify signatures of observations, actions and rewards

   # Option 1: Initialize with Gym environment
   import gym
   env = gym.make("Pendulum-v1")
   replay_buffer = d3rlpy.dataset.ReplayBuffer(
      buffer=buffer,
      transition_picker=transition_picker,
      trajectory_slicer=trajectory_slicer,
      writer_preprocessor=writer_preprocessor,
      env=env,
   )

   # Option 2: Initialize with pre-collected dataset
   dataset, _ = d3rlpy.datasets.get_pendulum()
   replay_buffer = d3rlpy.dataset.ReplayBuffer(
      buffer=buffer,
      transition_picker=transition_picker,
      trajectory_slicer=trajectory_slicer,
      writer_preprocessor=writer_preprocessor,
      episodes=dataset.episodes,
   )

   # Option 3: Initialize with manually specified signatures
   observation_signature = d3rlpy.dataset.Signature(shape=[(3,)], dtype=[np.float32])
   action_signature = d3rlpy.dataset.Signature(shape=[(1,)], dtype=[np.float32])
   reward_signature = d3rlpy.dataset.Signature(shape=[(1,)], dtype=[np.float32])
   replay_buffer = d3rlpy.dataset.ReplayBuffer(
      buffer=buffer,
      transition_picker=transition_picker,
      trajectory_slicer=trajectory_slicer,
      writer_preprocessor=writer_preprocessor,
      observation_signature=observation_signature,
      action_signature=action_signature,
      reward_signature=reward_signature,
   )

   # shortcut
   replay_buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.ReplayBuffer
   d3rlpy.dataset.create_infinite_replay_buffer
   d3rlpy.dataset.create_fifo_replay_buffer


Buffer
~~~~~~

``Buffer`` is a list-like component that stores and drops transitions.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.BufferProtocol
   d3rlpy.dataset.InfiniteBuffer
   d3rlpy.dataset.FIFOBuffer



TransitionPicker
~~~~~~~~~~~~~~~~

``TransitionPicker`` is a component that defines how to pick transition data used for Q-learning-based algorithms.
You can also implement your own ``TransitionPicker`` for custom experiments.

.. code-block:: python

   import d3rlpy

   # Example TransitionPicker that simply picks transition
   class CustomTransitionPicker(d3rlpy.dataset.TransitionPickerProtocol):
       def __call__(self, episode: d3rlpy.dataset.EpisodeBase, index: int) -> d3rlpy.dataset.Transition:
          observation = episode.observations[index]
          is_terminal = episode.terminated and index == episode.size() - 1
          if is_terminal:
              next_observation = d3rlpy.dataset.create_zero_observation(observation)
          else:
              next_observation = episode.observations[index + 1]
          return d3rlpy.dataset.Transition(
              observation=observation,
              action=episode.actions[index],
              reward=episode.rewards[index],
              next_observation=next_observation,
              terminal=float(is_terminal),
              interval=1,
          )


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.TransitionPickerProtocol
   d3rlpy.dataset.BasicTransitionPicker
   d3rlpy.dataset.FrameStackTransitionPicker
   d3rlpy.dataset.MultiStepTransitionPicker


TrajectorySlicer
~~~~~~~~~~~~~~~~

``TrajectorySlicer`` is a component that defines how to slice trajectory data used for Decision Transformer-based algorithms.
You can also implement your own ``TrajectorySlicer`` for custom experiments.

.. code-block:: python

   import d3rlpy

   class CustomTrajectorySlicer(d3rlpy.dataset.TrajectorySlicerProtocol):
       def __call__(
           self, episode: d3rlpy.dataset.EpisodeBase, end_index: int, size: int
       ) -> d3rlpy.dataset.PartialTrajectory:
           end = end_index + 1
           start = max(end - size, 0)
           actual_size = end - start

           # prepare terminal flags
           terminals = np.zeros((actual_size, 1), dtype=np.float32)
           if episode.terminated and end_index == episode.size() - 1:
               terminals[-1][0] = 1.0

           # slice data
           observations = episode.observations[start:end]
           actions = episode.actions[start:end]
           rewards = episode.rewards[start:end]
           ret = np.sum(episode.rewards[start:])
           all_returns_to_go = ret - np.cumsum(episode.rewards[start:], axis=0)
           returns_to_go = all_returns_to_go[:actual_size].reshape((-1, 1))

           # prepare metadata
           timesteps = np.arange(start, end)
           masks = np.ones(end - start, dtype=np.float32)

           # compute backward padding size
           pad_size = size - actual_size

           if pad_size == 0:
               return d3rlpy.dataset.PartialTrajectory(
                   observations=observations,
                   actions=actions,
                   rewards=rewards,
                   returns_to_go=returns_to_go,
                   terminals=terminals,
                   timesteps=timesteps,
                   masks=masks,
                   length=size,
               )

           return d3rlpy.dataset.PartialTrajectory(
               observations=d3rlpy.dataset.batch_pad_observations(observations, pad_size),
               actions=d3rlpy.dataset.batch_pad_array(actions, pad_size),
               rewards=d3rlpy.dataset.batch_pad_array(rewards, pad_size),
               returns_to_go=d3rlpy.dataset.batch_pad_array(returns_to_go, pad_size),
               terminals=d3rlpy.dataset.batch_pad_array(terminals, pad_size),
               timesteps=d3rlpy.dataset.batch_pad_array(timesteps, pad_size),
               masks=d3rlpy.dataset.batch_pad_array(masks, pad_size),
               length=size,
           )


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.TrajectorySlicerProtocol
   d3rlpy.dataset.BasicTrajectorySlicer



WriterPreprocess
~~~~~~~~~~~~~~~~

``WriterPreprocess`` is a component that defines how to write experiences to an experience replay buffer.
You can also implement your own ``WriterPreprocess`` for custom experiments.


.. code-block:: python

   import d3rlpy

   class CustomWriterPreprocess(d3rlpy.dataset.WriterPreprocessProtocol):
       def process_observation(self, observation: d3rlpy.dataset.Observation) -> d3rlpy.dataset.Observation:
           return observation

       def process_action(self, action: np.ndarray) -> np.ndarray:
           return action

       def process_reward(self, reward: np.ndarray) -> np.ndarray:
           return reward



.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.WriterPreprocessProtocol
   d3rlpy.dataset.BasicWriterPreprocess
   d3rlpy.dataset.LastFrameWriterPreprocess
