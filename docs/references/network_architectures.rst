Network Architectures
=====================

.. module:: d3rlpy.models

In d3rlpy, the neural network architecture is automatically selected based on
observation shape.
If the observation is image, the algorithm uses the ``Nature DQN``-based
encoder at each function.
Otherwise, the standard MLP architecture that consists with two linear
layers with ``256`` hidden units.

Furthermore, d3rlpy provides ``EncoderFactory`` that gives you flexible control
over this neural netowrk architectures.

.. code-block:: python

   import d3rlpy

   # encoder factory
   encoder_factory = d3rlpy.models.VectorEncoderFactory(
       hidden_units=[300, 400],
       activation='tanh',
   )

   # set EncoderFactory
   dqn = d3rlpy.algos.DQNConfig(encoder_factory=encoder_factory).create()

You can also build your own encoder factory.

.. code-block:: python

   import dataclasses
   import torch
   import torch.nn as nn

   from d3rlpy.models.encoders import EncoderFactory

   # your own neural network
   class CustomEncoder(nn.Module):
       def __init__(self, obsevation_shape, feature_size):
           self.feature_size = feature_size
           self.fc1 = nn.Linear(observation_shape[0], 64)
           self.fc2 = nn.Linear(64, feature_size)

       def forward(self, x):
           h = torch.relu(self.fc1(x))
           h = torch.relu(self.fc2(h))
           return h

   # your own encoder factory
   @dataclasses.dataclass()
   class CustomEncoderFactory(EncoderFactory):
       feature_size: int

       def create(self, observation_shape):
           return CustomEncoder(observation_shape, self.feature_size)

       @staticmethod
       def get_type() -> str:
           return "custom"

   dqn = d3rlpy.algos.DQNConfig(
      encoder_factory=CustomEncoderFactory(feature_size=64),
   ).create()


You can also define action-conditioned networks such as Q-functions for continuous
controls.
``create`` or ``create_with_action`` will be called depending on the function.

.. code-block:: python

   class CustomEncoderWithAction(nn.Module):
       def __init__(self, obsevation_shape, action_size, feature_size):
           self.feature_size = feature_size
           self.fc1 = nn.Linear(observation_shape[0] + action_size, 64)
           self.fc2 = nn.Linear(64, feature_size)

       def forward(self, x, action): # action is also given
           h = torch.cat([x, action], dim=1)
           h = torch.relu(self.fc1(h))
           h = torch.relu(self.fc2(h))
           return h

   @dataclasses.dataclass()
   class CustomEncoderFactory(EncoderFactory):
       feature_size: int

       def create(self, observation_shape):
           return CustomEncoder(observation_shape, self.feature_size)

       def create_with_action(observation_shape, action_size, discrete_action):
           return CustomEncoderWithAction(observation_shape, action_size, self.feature_size)

       @staticmethod
       def get_type() -> str:
           return "custom"


   factory = CustomEncoderFactory(feature_size=64)

   sac = d3rlpy.algos.SACConfig(
      actor_encoder_factory=factory,
      critic_encoder_factory=factory,
   ).create()

If you want ``load_learnable`` method to load the algorithm configuration including
your encoder configuration, you need to register your encoder factory.

.. code-block:: python

   from d3rlpy.models.encoders import register_encoder_factory

   # register your own encoder factory
   register_encoder_factory(CustomEncoderFactory)

   # load algorithm from d3
   dqn = d3rlpy.load_learnable("model.d3")


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.models.DefaultEncoderFactory
   d3rlpy.models.PixelEncoderFactory
   d3rlpy.models.VectorEncoderFactory
