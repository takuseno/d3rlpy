Command Line Interface
======================

d3rlpy provides the convenient CLI tool.

plot
----

Plot the saved metrics by specifying paths::

  $ d3rlpy plot <path> [<path>...]

.. list-table:: options
   :header-rows: 1

   * - option
     - description
   * - ``--window``
     - moving average window.
   * - ``--show-steps``
     - use iterations on x-axis.
   * - ``--show-max``
     - show maximum value.

example::

  $ d3rlpy plot d3rlpy_logs/CQL_20201224224314/environment.csv

.. image:: ./assets/plot_example.png

plot-all
--------

Plot the all metrics saved in the directory::

  $ d3rlpy plot-all <path>

example::

  $ d3rlpy plot d3rlpy_logs/CQL_20201224224314

.. image:: ./assets/plot_all_example.png

export
------

Export the saved model to the inference format, ``onnx`` and ``torchscript``::

  $ d3rlpy export <path>

.. list-table:: options
   :header-rows: 1

   * - option
     - description
   * - ``--format``
     - model format (torchscript, onnx).
   * - ``--params-json``
     - explicitly specify params.json.
   * - ``--out``
     - output path.

example::

  $ d3rlpy export d3rlpy_logs/CQL_20201224224314/model_100.pt

