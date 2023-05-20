Preprocessing
=============

.. module:: d3rlpy.preprocessing

Observation
~~~~~~~~~~~

d3rlpy provides several preprocessors tightly incorporated with algorithms.
Each preprocessor is implemented with PyTorch operation, which will be included
in the model exported by `save_policy` method.

.. code-block:: python

    from d3rlpy.datasets import get_pendulum
    from d3rlpy.algos import CQLConfig
    from d3rlpy.preprocesing import StandardObservationScaler

    dataset, _ = get_pendulum()

    # choose from ['pixel', 'min_max', 'standard'] or None
    cql = CQLConfig(observation_scaler=StandardObservationScaler()).create()

    # observation scaler is fitted from the given dataset
    cql.fit(dataset, n_steps=100000)

    # preprocesing is included in TorchScript
    cql.save_policy('policy.pt')

    # you don't need to take care of preprocessing at production
    policy = torch.jit.load('policy.pt')
    action = policy(unpreprocessed_x)

You can also initialize observation scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import StandardObservationScaler

    observation_scaler = StandardObservationScaler(mean=..., std=...)

    cql = CQLConfig(observation_scaler=observation_scaler).create()

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.PixelObservationScaler
   d3rlpy.preprocessing.MinMaxObservationScaler
   d3rlpy.preprocessing.StandardObservationScaler


Action
~~~~~~

d3rlpy also provides the feature that preprocesses continuous action.
With this preprocessing, you don't need to normalize actions in advance or
implement normalization in the environment side.

.. code-block:: python

    from d3rlpy.datasets import get_pendulum
    from d3rlpy.algos import CQLConfig
    from d3rlpy.preprocessing import MinMaxActionScaler

    dataset, _ = get_pendulum()

    cql = CQLConfig(action_scaler=MinMaxActionScaler()).create()

    # action scaler is fitted from the given episodes
    cql.fit(dataset, n_steps=100000)

    # postprocessing is included in TorchScript
    cql.save_policy('policy.pt')

    # you don't need to take care of postprocessing at production
    policy = torch.jit.load('policy.pt')
    action = policy(x)

You can also initialize scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import MinMaxActionScaler

    action_scaler = MinMaxActionScaler(minimum=..., maximum=...)

    cql = CQLConfig(action_scaler=action_scaler).create()

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.MinMaxActionScaler


Reward
~~~~~~

d3rlpy also provides the feature that preprocesses rewards.
With this preprocessing, you don't need to normalize rewards in advance.
Note that this preprocessor should be fitted with the dataset.
Afterwards you can use it with online training.

.. code-block:: python

    from d3rlpy.datasets import get_pendulum
    from d3rlpy.algos import CQLConfig
    from d3rlpy.preprocessing import StandardRewardScaler

    dataset, _ = get_pendulum()

    cql = CQLConfig(reward_scaler=StandardRewardScaler()).create()

    # reward scaler is fitted from the given episodes
    cql.fit(dataset)

    # reward scaler is also available at finetuning.
    cql.fit_online(env)

You can also initialize scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import MinMaxRewardScaler

    reward_scaler = MinMaxRewardScaler(minimum=..., maximum=...)

    cql = CQLConfig(reward_scaler=reward_scaler).create()

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.MinMaxRewardScaler
   d3rlpy.preprocessing.StandardRewardScaler
   d3rlpy.preprocessing.ClipRewardScaler
   d3rlpy.preprocessing.MultiplyRewardScaler
   d3rlpy.preprocessing.ReturnBasedRewardScaler
   d3rlpy.preprocessing.ConstantShiftRewardScaler
