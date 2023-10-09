Algorithms
==========

.. module:: d3rlpy.algos

d3rlpy provides state-of-the-art offline deep reinforcement
learning algorithms as well as online algorithms for the base implementations.

Each algorithm provides its config class and you can instantiate it with specifying a device to use.

.. code-block:: python

   import d3rlpy

   # instantiate algorithm with CPU
   sac = d3rlpy.algos.SACConfig().create(device="cpu:0")
   # instantiate algorithm with GPU
   sac = d3rlpy.algos.SACConfig().create(device="cuda:0")
   # instantiate algorithm with the 2nd GPU
   sac = d3rlpy.algos.SACConfig().create(device="cuda:1")


You can also check advanced use cases at `examples <https://github.com/takuseno/d3rlpy/tree/master/examples>`_ directory.


Base
~~~~

LearnableBase
-------------

The base class of all algorithms.

.. autoclass:: d3rlpy.base.LearnableBase
   :members:
   :show-inheritance:



Q-learning
~~~~~~~~~~

QLearningAlgoBase
-----------------

The base class of Q-learning algorithms.

.. autoclass:: d3rlpy.algos.QLearningAlgoBase
   :members:
   :show-inheritance:



BC
--

.. autoclass:: d3rlpy.algos.BCConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.BC
   :members:
   :show-inheritance:

DiscreteBC
----------

.. autoclass:: d3rlpy.algos.DiscreteBCConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteBC
   :members:
   :show-inheritance:

NFQ
---

.. autoclass:: d3rlpy.algos.NFQConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.NFQ
   :members:
   :show-inheritance:


DQN
---

.. autoclass:: d3rlpy.algos.DQNConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DQN
   :members:
   :show-inheritance:


DoubleDQN
---------

.. autoclass:: d3rlpy.algos.DoubleDQNConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DoubleDQN
   :members:
   :show-inheritance:


DDPG
----

.. autoclass:: d3rlpy.algos.DDPGConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DDPG
   :members:
   :show-inheritance:


TD3
---

.. autoclass:: d3rlpy.algos.TD3Config
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.TD3
   :members:
   :show-inheritance:


SAC
---

.. autoclass:: d3rlpy.algos.SACConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.SAC
   :members:
   :show-inheritance:


DiscreteSAC
-----------

.. autoclass:: d3rlpy.algos.DiscreteSACConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteSAC
   :members:
   :show-inheritance:


BCQ
---

.. autoclass:: d3rlpy.algos.BCQConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.BCQ
   :members:
   :show-inheritance:


DiscreteBCQ
-----------

.. autoclass:: d3rlpy.algos.DiscreteBCQConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteBCQ
   :members:
   :show-inheritance:


BEAR
----

.. autoclass:: d3rlpy.algos.BEARConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.BEAR
   :members:
   :show-inheritance:


CRR
---

.. autoclass:: d3rlpy.algos.CRRConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.CRR
   :members:
   :show-inheritance:


CQL
---

.. autoclass:: d3rlpy.algos.CQLConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.CQL
   :members:
   :show-inheritance:


DiscreteCQL
-----------

.. autoclass:: d3rlpy.algos.DiscreteCQLConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteCQL
   :members:
   :show-inheritance:


AWAC
----

.. autoclass:: d3rlpy.algos.AWACConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.AWAC
   :members:
   :show-inheritance:


PLAS
----

.. autoclass:: d3rlpy.algos.PLASConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.PLAS
   :members:
   :show-inheritance:


PLAS+P
------

.. autoclass:: d3rlpy.algos.PLASWithPerturbationConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.PLASWithPerturbation
   :members:
   :show-inheritance:


TD3+BC
------

.. autoclass:: d3rlpy.algos.TD3PlusBCConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.TD3PlusBC
   :members:
   :show-inheritance:


IQL
---

.. autoclass:: d3rlpy.algos.IQLConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.IQL
   :members:
   :show-inheritance:


RandomPolicy
------------

.. autoclass:: d3rlpy.algos.RandomPolicyConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.RandomPolicy
   :members:
   :show-inheritance:


DiscreteRandomPolicy
--------------------

.. autoclass:: d3rlpy.algos.DiscreteRandomPolicyConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteRandomPolicy
   :members:
   :show-inheritance:


Decision Transformer
~~~~~~~~~~~~~~~~~~~~

Decision Transformer-based algorithms usually require tricky interaction codes for evaluation.
In d3rlpy, those algorithms provide ``as_stateful_wrapper`` method to easily integrate the algorithm into your system.

.. code-block:: python

   import d3rlpy

   dataset, env = d3rlpy.datasets.get_pendulum()

   dt = d3rlpy.algos.DecisionTransformerConfig().create(device="cuda:0")

   # offline training
   dt.fit(
      dataset,
      n_steps=100000,
      n_steps_per_epoch=1000,
      eval_env=env,
      eval_target_return=0,  # specify target environment return
   )

   # wrap as stateful actor for interaction
   actor = dt.as_stateful_wrapper(target_return=0)

   # interaction
   observation, reward = env.reset(), 0.0
   while True:
       action = actor.predict(observation, reward)
       observation, reward, done, truncated, _ = env.step(action)
       if done or truncated:
           break

   # reset history
   actor.reset()


TransformerAlgoBase
-------------------

.. autoclass:: d3rlpy.algos.TransformerAlgoBase
   :members:
   :show-inheritance:


DecisionTransformer
--------------------

.. autoclass:: d3rlpy.algos.DecisionTransformerConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DecisionTransformer
   :members:
   :show-inheritance:


DiscreteDecisionTransformer
---------------------------

.. autoclass:: d3rlpy.algos.DiscreteDecisionTransformerConfig
   :members:
   :show-inheritance:

.. autoclass:: d3rlpy.algos.DiscreteDecisionTransformer
   :members:
   :show-inheritance:


TransformerActionSampler
------------------------

``TransformerActionSampler`` is an interface to sample actions from
DecisionTransformer outputs. Basically, the default action-sampler will be used
if you don't explicitly specify one.

.. code-block:: python

   import d3rlpy

   dataset, env = d3rlpy.datasets.get_pendulum()

   dt = d3rlpy.algos.DecisionTransformerConfig().create(device="cuda:0")

   # offline training
   dt.fit(
      dataset,
      n_steps=100000,
      n_steps_per_epoch=1000,
      eval_env=env,
      eval_target_return=0,
      # manually specify action-sampler
      eval_action_sampler=d3rlpy.algos.IdentityTransformerActionSampler(),
   )

   # wrap as stateful actor for interaction with manually specified action-sampler
   actor = dt.as_stateful_wrapper(
       target_return=0,
       action_sampler=d3rlpy.algos.IdentityTransformerActionSampler(),
   )


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.algos.TransformerActionSampler
   d3rlpy.algos.IdentityTransformerActionSampler(
   d3rlpy.algos.SoftmaxTransformerActionSampler
   d3rlpy.algos.GreedyTransformerActionSampler
