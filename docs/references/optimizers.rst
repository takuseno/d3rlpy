Optimizers
==========

.. module:: d3rlpy.models

d3rlpy provides ``OptimizerFactory`` that gives you flexible control over
optimizers.
``OptimizerFactory`` takes PyTorch's optimizer class and its arguments to
initialize, which you can check more `here <https://pytorch.org/docs/stable/optim.html>`_.

.. code-block:: python

   import d3rlpy
   from torch.optim import Adam

   # modify weight decay
   optim_factory = d3rlpy.models.OptimizerFactory(Adam, weight_decay=1e-4)

   # set OptimizerFactory
   dqn = d3rlpy.algos.DQNConfig(optim_factory=optim_factory).create()

There are also convenient alises.

.. code-block:: python

   # alias for Adam optimizer
   optim_factory = d3rlpy.models.AdamFactory(weight_decay=1e-4)

   dqn = d3rlpy.algos.DQNConfig(optim_factory=optim_factory).create()


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.models.OptimizerFactory
   d3rlpy.models.SGDFactory
   d3rlpy.models.AdamFactory
   d3rlpy.models.RMSpropFactory
   d3rlpy.models.GPTAdamWFactory


Learning rate scheduler
~~~~~~~~~~~~~~~~~~~~~~~

d3rlpy provides ``LRSchedulerFactory`` that gives you configure learning rate
schedulers with ``OptimizerFactory``.

.. code-block:: python

   import d3rlpy

   # set lr_scheduler_factory
   optim_factory = d3rlpy.models.AdamFactory(
       lr_scheduler_factory=d3rlpy.models.WarmupSchedulerFactory(
           warmup_steps=10000
       )
   )


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.models.LRSchedulerFactory
   d3rlpy.models.WarmupSchedulerFactory
   d3rlpy.models.CosineAnnealingLRFactory
