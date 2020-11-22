Optimizers
==========

.. module:: d3rlpy.optimizers

d3rlpy provides ``OptimizerFactory`` that gives you flexible control over
optimizers.
``OptimizerFactory`` takes PyTorch's optimizer class and its arguments to
initialize, which you can check more `here <https://pytorch.org/docs/stable/optim.html>`_.

.. code-block:: python

   from torch.optim import Adam
   from d3rlpy.algos import DQN
   from d3rlpy.optimizers import OptimizerFactory

   # modify weight decay
   optim_factory = OptimizerFactory(Adam, weight_decay=1e-4)

   # set OptimizerFactory
   dqn = DQN(optim_factory=optim_factory)

There are also convenient alises.

.. code-block:: python

   from d3rlpy.optimizers import AdamFactory

   # alias for Adam optimizer
   optim_factory = AdamFactory(weight_decay=1e-4)

   dqn = DQN(optim_factory=optim_factory)


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.optimizers.OptimizerFactory
   d3rlpy.optimizers.SGDFactory
   d3rlpy.optimizers.AdamFactory
   d3rlpy.optimizers.RMSpropFactory
