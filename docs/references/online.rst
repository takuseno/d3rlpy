Online Training
===============

.. module:: d3rlpy.algos

d3rlpy provides not only offline training, but also online training utilities.
Despite being designed for offline training algorithms, d3rlpy is flexible
enough to be trained in an online manner with a few more utilities.

.. code-block:: python

    import d3lpy
    import gym

    # setup environment
    env = gym.make('CartPole-v1')
    eval_env = gym.make('CartPole-v1')

    # setup algorithm
    dqn = d3rlpy.algos.DQN(
        batch_size=32,
        learning_rate=2.5e-4,
        target_update_interval=100,
    ).create(device="cuda:0")

    # setup replay buffer
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(limit=100000, env=env)

    # setup explorers
    explorer = d3rlpy.algos.LinearDecayEpsilonGreedy(
        start_epsilon=1.0,
        end_epsilon=0.1,
        duration=10000,
    )

    # start training
    dqn.fit_online(
        env,
        buffer,
        explorer=explorer, # you don't need this with probablistic policy algorithms
        eval_env=eval_env,
        n_steps=30000, # the number of total steps to train.
        n_steps_per_epoch=1000,
        update_interval=10, # update parameters every 10 steps.
    )


Explorers
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.algos.ConstantEpsilonGreedy
   d3rlpy.algos.LinearDecayEpsilonGreedy
   d3rlpy.algos.NormalNoise
