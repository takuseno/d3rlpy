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
    from d3rlpy.preprocessing import StandardScaler

    dataset = MDPDataset(...)

    # initialize scaler
    scaler = StandardScaler()

    # pass scaler to algorithm
    cql = CQL(scaler=scaler)

    # scaler is fitted from the given episodes
    cql.fit(dataset.episodes)

    # preprocesing is included in TorchScript
    cql.save_policy('policy.pt')

    # you don't need to take care of preprocessing at production
    policy = torch.jit.load('policy.pt')
    action = policy(unpreprocessed_x)

You can also make your own preprocessors.

.. code-block:: python

    from d3rlpy.preprocessing import Scaler

    class LogScaler(Scaler):
        def fit(self, episodes):
            pass

        def transform(self, x):
            return x.log()

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.preprocessing.PixelScaler
   d3rlpy.preprocessing.MinMaxScaler
   d3rlpy.preprocessing.StandardScaler
