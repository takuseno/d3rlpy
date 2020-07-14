Preprocessing
=============

.. module:: d3rlpy.preprocessing

d3rlpy provides several preprocessors tightly incorporated with
algorithms.
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
