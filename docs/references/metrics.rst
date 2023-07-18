Metrics
=======

.. module:: d3rlpy.metrics

d3rlpy provides scoring functions for offline Q-learning-based training.
You can also check :doc:`../references/logging` to understand how to write
metrics to files.

.. code-block:: python

    import d3rlpy

    dataset, env = d3rlpy.datasets.get_cartpole()
    # use partial episodes as test data
    test_episodes = dataset.episodes[:10]

    dqn = d3rlpy.algos.DQNConfig().create()

    dqn.fit(
        dataset,
        n_steps=100000,
        evaluators={
            'td_error': d3rlpy.metrics.TDErrorEvaluator(test_episodes),
            'value_scale': d3rlpy.metrics.AverageValueEstimationEvaluator(test_episodes),
            'environment': d3rlpy.metrics.EnvironmentEvaluator(env),
        },
    )

You can also implement your own metrics.


.. code-block:: python

    class CustomEvaluator(d3rlpy.metrics.EvaluatorProtocol):
        def __call__(self, algo: d3rlpy.algos.QLearningAlgoBase, dataset: ReplayBuffer) -> float:
            # do some evaluation


.. autosummary::
   :toctree: generated/
   :nosignatures:

   d3rlpy.metrics.TDErrorEvaluator
   d3rlpy.metrics.DiscountedSumOfAdvantageEvaluator
   d3rlpy.metrics.AverageValueEstimationEvaluator
   d3rlpy.metrics.InitialStateValueEstimationEvaluator
   d3rlpy.metrics.SoftOPCEvaluator
   d3rlpy.metrics.ContinuousActionDiffEvaluator
   d3rlpy.metrics.DiscreteActionMatchEvaluator
   d3rlpy.metrics.EnvironmentEvaluator
   d3rlpy.metrics.CompareContinuousActionDiffEvaluator
   d3rlpy.metrics.CompareDiscreteActionMatchEvaluator
