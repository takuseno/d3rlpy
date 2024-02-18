.. _logging:

Logging
=======

.. module:: d3rlpy.logging

d3rlpy provides a customizable interface for logging metrics, ``LoggerAdapter`` and ``LoggerAdapterFactory``.

.. code-block:: python

   import d3rlpy

   dataset, env = d3rlpy.datasets.get_cartpole()

   dqn = d3rlpy.algos.DQNConfig().create()

   dqn.fit(
      dataset=dataset,
      n_steps=100000,
      # set FileAdapterFactory to save metrics as CSV files
      logger_adapter=d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
   )

``LoggerAdapterFactory`` is a parent interface that instantiates ``LoggerAdapter`` at the beginning of training.
You can also use ``CombineAdapter`` to combine multiple ``LoggerAdapter`` in the same training.

.. code-block:: python

   # combine FileAdapterFactory and TensorboardAdapterFactory
   logger_adapter = d3rlpy.logging.CombineAdapterFactory([
      d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
      d3rlpy.logging.TensorboardAdapterFactory(root_dir="tensorboard_logs"),
   ])

   dqn.fit(dataset=dataset, n_steps=100000, logger_adapter=logger_adapter)


LoggerAdapter
-------------

``LoggerAdapter`` is an inner interface of ``LoggerAdapterFactory``.
You can implement your own ``LoggerAdapter`` for 3rd-party visualizers.


.. code-block:: python

   import d3rlpy

   class CustomAdapter(d3rlpy.logging.LoggerAdapter):
       def write_params(self, params: Dict[str, Any]) -> None:
           # save dictionary as json file
           with open("params.json", "w") as f:
               f.write(json.dumps(params, default=default_json_encoder, indent=2))

       def before_write_metric(self, epoch: int, step: int) -> None:
           pass

       def write_metric(
           self, epoch: int, step: int, name: str, value: float
       ) -> None:
           with open(f"{name}.csv", "a") as f:
               print(f"{epoch},{step},{value}", file=f)

       def after_write_metric(self, epoch: int, step: int) -> None:
           pass

       def save_model(self, epoch: int, algo: Any) -> None:
           algo.save(f"model_{epoch}.d3")

       def close(self) -> None:
           pass

.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.logging.LoggerAdapter
   d3rlpy.logging.FileAdapter
   d3rlpy.logging.TensorboardAdapter
   d3rlpy.logging.WanDBAdapter
   d3rlpy.logging.NoopAdapter
   d3rlpy.logging.CombineAdapter

LoggerAdapterFactory
--------------------

``LoggerAdapterFactory`` is an interface that instantiates ``LoggerAdapter`` at the beginning of training.
You can implement your own ``LoggerAdapterFactory`` for 3rd-party visualizers.

.. code-block:: python

   import d3rlpy

   class CustomAdapterFactory(d3rlpy.logging.LoggerAdapterFactory):
       def create(self, experiment_name: str) -> d3rlpy.logging.FileAdapter:
           return CustomAdapter()


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.logging.LoggerAdapterFactory
   d3rlpy.logging.FileAdapterFactory
   d3rlpy.logging.TensorboardAdapterFactory
   d3rlpy.logging.WanDBAdapterFactory
   d3rlpy.logging.NoopAdapterFactory
   d3rlpy.logging.CombineAdapterFactory
