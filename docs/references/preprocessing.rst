Preprocessing
=============

.. module:: d3rlpy.preprocessing

Observation
~~~~~~~~~~~

d3rlpy provides several preprocessors tightly incorporated with algorithms.
Each preprocessor is implemented with PyTorch operation, which will be included
in the model exported by `save_policy` method.

.. code-block:: python

    from d3rlpy.algos import CQL
    from d3rlpy.dataset import MDPDataset

    dataset = MDPDataset(...)

    # choose from ['pixel', 'min_max', 'standard'] or None
    cql = CQL(scaler='standard')

    # scaler is fitted from the given episodes
    cql.fit(dataset.episodes)

    # preprocesing is included in TorchScript
    cql.save_policy('policy.pt')

    # you don't need to take care of preprocessing at production
    policy = torch.jit.load('policy.pt')
    action = policy(unpreprocessed_x)

You can also initialize scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import StandardScaler

    scaler = StandardScaler(mean=..., std=...)

    cql = CQL(scaler=scaler)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.PixelScaler
   d3rlpy.preprocessing.MinMaxScaler
   d3rlpy.preprocessing.StandardScaler


Action
~~~~~~

d3rlpy also provides the feature that preprocesses continuous action.
With this preprocessing, you don't need to normalize actions in advance or
implement normalization in the environment side.

.. code-block:: python

    from d3rlpy.algos import CQL
    from d3rlpy.dataset import MDPDataset

    dataset = MDPDataset(...)

    # 'min_max' or None
    cql = CQL(action_scaler='min_max')

    # action scaler is fitted from the given episodes
    cql.fit(dataset.episodes)

    # postprocessing is included in TorchScript
    cql.save_policy('policy.pt')

    # you don't need to take care of postprocessing at production
    policy = torch.jit.load('policy.pt')
    action = policy(x)

You can also initialize scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import MinMaxActionScaler

    action_scaler = MinMaxActionScaler(minimum=..., maximum=...)

    cql = CQL(action_scaler=action_scaler)

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

    from d3rlpy.algos import CQL
    from d3rlpy.dataset import MDPDataset

    dataset = MDPDataset(...)

    # 'min_max', 'standard' or None
    cql = CQL(reward_scaler='standard')

    # reward scaler is fitted from the given episodes
    cql.fit(dataset.episodes)

    # reward scaler is also available at finetuning.
    cql.fit_online(env)

You can also initialize scalers by yourself.

.. code-block:: python

    from d3rlpy.preprocessing import MinMaxRewardScaler

    reward_scaler = MinMaxRewardScaler(minimum=..., maximum=...)

    cql = CQL(reward_scaler=reward_scaler)

    # ClipRewardScaler and MultiplyRewardScaler must be initialized manually
    reward_scaler = ClipRewardScaler(-1.0, 1.0)
    cql = CQL(reward_scaler=reward_scaler)

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.MinMaxRewardScaler
   d3rlpy.preprocessing.StandardRewardScaler
   d3rlpy.preprocessing.ClipRewardScaler
   d3rlpy.preprocessing.MultiplyRewardScaler
