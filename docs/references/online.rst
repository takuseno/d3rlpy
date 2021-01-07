Online Training
===============

.. module:: d3rlpy.online

d3rlpy provides not only offline training, but also online training utilities.
Despite being designed for offline training algorithms, d3rlpy is flexible
enough to be trained in an online manner with a few more utilities.

.. code-block:: python

    import gym

    from d3rlpy.algos import DQN
    from d3rlpy.online.buffers import ReplayBuffer
    from d3rlpy.online.explorers import LinearDecayEpsilonGreedy

    # setup environment
    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')

    # setup algorithm
    dqn = DQN(batch_size=32,
              learning_rate=2.5e-4,
              target_update_interval=100,
              use_gpu=True)

    # setup replay buffer
    buffer = ReplayBuffer(maxlen=1000000, env=env)

    # setup explorers
    explorer = LinearDecayEpsilonGreedy(start_epsilon=1.0,
                                        end_epsilon=0.1,
                                        duration=10000)

    # start training
    dqn.fit_online(env,
                   buffer,
                   explorer=explorer, # you don't need this with probablistic policy algorithms
                   eval_env=eval_env,
                   n_epochs=30,
                   n_steps_per_epoch=1000,
                   n_updates_per_epoch=100)


Replay Buffer
-------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.online.buffers.ReplayBuffer


Explorers
---------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.online.explorers.LinearDecayEpsilonGreedy
   d3rlpy.online.explorers.NormalNoise
