************************
Customize Neural Network
************************

In this tutorial, you can learn how to integrate your own neural network models
to d3rlpy.
Please check :doc:`../references/network_architectures` for more information.

Prepare PyTorch Model
---------------------

If you're familiar with PyTorch, this step should be easy for you.
Please note that your model must have ``get_feature_size`` method to tell the
feature size to the final layer.

.. code-block:: python

  import torch
  import torch.nn as nn
  import d3rlpy

  class CustomEncoder(nn.Module):
      def __init__(self, observation_shape, feature_size):
          super().__init__()
          self.feature_size = feature_size
          self.fc1 = nn.Linear(observation_shape[0], feature_size)
          self.fc2 = nn.Linear(feature_size, feature_size)

      def forward(self, x):
          h = torch.relu(self.fc1(x))
          h = torch.relu(self.fc2(h))
          return h

      # THIS IS IMPORTANT!
      def get_feature_size(self):
          return self.feature_size

Setup EncoderFactory
--------------------

Once you setup your PyTorch model, you need to setup ``EncoderFactory``.
In your ``EncoderFactory`` class, you need to define ``create`` and
``get_params`` methods as well as ``TYPE`` attribute.
``TYPE`` attribute and ``get_params`` method are used to serialize your
customized neural network configuration.

.. code-block:: python

  class CustomEncoderFactory(d3rlpy.models.encoders.EncoderFactory):
      TYPE = "custom"  # this is necessary

      def __init__(self, feature_size):
          self.feature_size = feature_size

      def create(self, observation_shape):
          return CustomEncoder(observation_shape, self.feature_size)

      def get_params(self, deep=False):
          return {"feature_size": self.feature_size}


Now, you can use your model with d3rlpy.

.. code-block:: python

  # integrate your model into d3rlpy algorithm
  dqn = d3rlpy.algos.DQNConfig(encoder_factory=CustomEncoderFactory(64)).create()


Support Q-function for Actor-Critic
-----------------------------------

In the above example, your original model is designed for the network that
takes an observation as an input.
However, if you customize a Q-function of actor-critic algorithm (e.g. SAC),
you need to prepare an action-conditioned model.

.. code-block:: python

  class CustomEncoderWithAction(nn.Module):
      def __init__(self, observation_shape, action_size, feature_size):
          super().__init__()
          self.feature_size = feature_size
          self.fc1 = nn.Linear(observation_shape[0] + action_size, feature_size)
          self.fc2 = nn.Linear(feature_size, feature_size)

        def forward(self, x, action):
            h = torch.cat([x, action], dim=1)
            h = torch.relu(self.fc1(h))
            h = torch.relu(self.fc2(h))
            return h

        def get_feature_size(self):
            return self.feature_size

Finally, you can update your ``CustomEncoderFactory`` as follows.

.. code-block:: python

  class CustomEncoderFactory(EncoderFactory):
      TYPE = "custom"

      def __init__(self, feature_size):
          self.feature_size = feature_size

      def create(self, observation_shape):
          return CustomEncoder(observation_shape, self.feature_size)

      def create_with_action(self, observation_shape, action_size, discrete_action):
          return CustomEncoderWithAction(observation_shape, action_size, self.feature_size)

      def get_params(self, deep=False):
          return {"feature_size": self.feature_size}

Now, you can customize actor-critic algorithms.

.. code-block:: python

  encoder_factory = CustomEncoderFactory(64)

  sac = d3rlpy.algos.SACConfig(
      actor_encoder_factory=encoder_factory,
      critic_encoder_factory=encoder_factory,
  ).create()
