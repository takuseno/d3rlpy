Network Architectures
=====================

.. module:: d3rlpy.encoders

In d3rlpy, the neural network architecture is automatically selected based on
observation shape.
If the observation is image, the algorithm uses the ``Nature DQN``-based
encoder at each function.
Otherwise, the standard MLP architecture that consists with two linear
layers with ``256`` hidden units.

Furthermore, d3rlpy provides ``EncoderFactory`` that gives you flexible control
over this neural netowrk architectures.

.. code-block:: python

   from d3rlpy.algos import DQN
   from d3rlpy.encoders import VectorEncoderFactory

   # encoder factory
   encoder_factory = VectorEncoderFactory(hidden_units=[300, 400],
                                          activation='tanh')

   # set OptimizerFactory
   dqn = DQN(encoder_factory=encoder_factory)

You can also build your own encoder factory.

.. code-block:: python

   import torch
   import torch.nn as nn

   from d3rlpy.encoders import EncoderFactory

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

       # THIS IS IMPORTANT!
       def get_feature_size(self):
           return self.feature_size

   # your own encoder factory
   class CustomEncoderFactory(EncoderFactory):
       TYPE = 'custom' # this is necessary

       def __init__(self, feature_size):
           self.feature_size = feature_size

       def create(self, observation_shape, action_size=None, discrete_action=False):
           return CustomEncoder(observation_shape, self.feature_size)

       def get_params(self, deep=False):
           return {
               'feature_size': self.feature_size
           }

   dqn = DQN(encoder_factory=CustomEncoderFactory(feature_size=64))


You can also share the factory across functions as below.

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

       def get_feature_size(self):
           return self.feature_size

   class CustomEncoderFactory(EncoderFactory):
       TYPE = 'custom' # this is necessary

       def __init__(self, feature_size):
           self.feature_size = feature_size

       def create(self, observation_shape, action_size=None, discrete_action=False):
           # branch based on if ``action_size`` is given.
           if action_size is None:
               return CustomEncoder(observation_shape, self.feature_size)
           else:
               return CustomEncoderWithAction(observation_shape,
                                              action_size,
                                              self.feature_size)

       def get_params(self, deep=False):
           return {
               'feature_size': self.feature_size
           }

   from d3rlpy.algos import SAC

   factory = CustomEncoderFactory(feature_size=64)

   sac = SAC(actor_encoder_factory=factory, critic_encoder_factory=factory)

If you want ``from_json`` method to load the algorithm configuration including
your encoder configuration, you need to register your encoder factory.

.. code-block:: python

   from d3rlpy.encoders import register_encoder_factory

   # register your own encoder factory
   register_encoder_factory(CustomEncoderFactory)

   # load algorithm from json
   dqn = DQN.from_json('<path-to-json>/params.json')

Once you register your encoder factory, you can specify it via ``TYPE`` value.

.. code-block:: python

   dqn = DQN(encoder_factory='custom')


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.encoders.DefaultEncoderFactory
   d3rlpy.encoders.PixelEncoderFactory
   d3rlpy.encoders.VectorEncoderFactory
