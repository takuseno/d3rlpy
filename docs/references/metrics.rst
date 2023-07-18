Metrics
=======

.. module:: d3rlpy.metrics

d3rlpy provides scoring functions without compromising scikit-learn
compatibility.
You can evaluate many metrics with test episodes during training.

.. code-block:: python

    from d3rlpy.datasets import get_cartpole
    from d3rlpy.algos import DQN
    from d3rlpy.metrics.scorer import td_error_scorer
    from d3rlpy.metrics.scorer import average_value_estimation_scorer
    from d3rlpy.metrics.scorer import evaluate_on_environment
    from sklearn.model_selection import train_test_split

    dataset, env = get_cartpole()

    train_episodes, test_episodes = train_test_split(dataset)

    dqn = DQN()

    dqn.fit(train_episodes,
            eval_episodes=test_episodes,
            scorers={
                'td_error': td_error_scorer,
                'value_scale': average_value_estimation_scorer,
                'environment': evaluate_on_environment(env)
            })

You can also use them with scikit-learn utilities.

.. code-block:: python

    from sklearn.model_selection import cross_validate

    scores = cross_validate(dqn,
                            dataset,
                            scoring={
                                'td_error': td_error_scorer,
                                'environment': evaluate_on_environment(env)
                            })


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
