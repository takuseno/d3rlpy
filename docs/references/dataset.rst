.. _mdp_dataset:

MDPDataset
==========

.. module:: d3rlpy.dataset

d3rlpy provides useful dataset structure for data-driven deep reinforcement
learning.
In supervised learning, the training script iterates input data :math:`X` and
label data :math:`Y`.
However, in reinforcement learning, mini-batches consist with sets of
:math:`(s_t, a_t, r_{t+1}, s_{t+1})` and episode terminal flags.
Converting a set of observations, actions, rewards and terminal flags into this
tuples is boring and requires some codings.

Therefore, d3rlpy provides `MDPDataset` class which enables you to handle
reinforcement learning datasets without any efforts.

.. code-block:: python

    from d3rlpy.dataset import MDPDataset

    # 1000 steps of observations with shape of (100,)
    observations = np.random.random((1000, 100))
    # 1000 steps of actions with shape of (4,)
    actions = np.random.random((1000, 4))
    # 1000 steps of rewards
    rewards = np.random.random(1000)
    # 1000 steps of terminal flags
    terminals = np.random.randint(2, size=1000)

    dataset = MDPDataset(observations, actions, rewards, terminals)

    # automatically splitted into d3rlpy.dataset.Episode objects
    dataset.episodes

    # each episode is also splitted into d3rlpy.dataset.Transition objects
    episode = dataset.episodes[0]
    episode[0].observation
    episode[0].action
    episode[0].next_reward
    episode[0].next_observation
    episode[0].terminal

    # d3rlpy.dataset.Transition object has pointers to previous and next
    # transitions like linked list.
    transition = episode[0]
    while transition.next_transition:
        transition = transition.next_transition

    # save as HDF5
    dataset.dump('dataset.h5')

    # load from HDF5
    new_dataset = MDPDataset.load('dataset.h5')

Please note that the ``observations``, ``actions``, ``rewards`` and ``terminals``
must be aligned with the same timestep.

.. code-block:: python

  observations = [s1, s2, s3, ...]
  actions      = [a1, a2, a3, ...]
  rewards      = [r1, r2, r3, ...]
  terminals    = [t1, t2, t3, ...]

This alignment might be different from other libraries where the tuple of :math:`(s_t, a_t, r_{t+1})`.
The advantage of d3rlpy's formulation is that we can explicitly store the last observation which might
be useful for the futrue goal-oriented methods and less confusing.

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.dataset.MDPDataset
   d3rlpy.dataset.Episode
   d3rlpy.dataset.Transition
   d3rlpy.dataset.TransitionMiniBatch
