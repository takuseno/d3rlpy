scikit-learn compatibility
==========================

d3rlpy provides complete scikit-learn compatible APIs.

train_test_split
----------------

:class:`d3rlpy.dataset.MDPDataset` is compatible with splitting functions in
scikit-learn.

.. code-block:: python

    from d3rlpy.algos import DQN
    from d3rlpy.datasets import get_cartpole
    from d3rlpy.metrics.scorer import td_error_scorer
    from sklearn.model_selection import train_test_split

    dataset, env = get_cartpole()

    train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

    dqn = DQN()
    dqn.fit(train_episodes,
            eval_episodes=test_episodes,
            n_epochs=1,
            scorers={'td_error': td_error_scorer})

cross_validate
--------------

cross validation is also easily performed.

.. code-block:: python

    from d3rlpy.algos import DQN
    from d3rlpy.datasets import get_cartpole
    from d3rlpy.metrics import td_error_scorer
    from sklearn.model_selection import cross_validate

    dataset, env = get_cartpole()

    dqn = DQN()

    scores = cross_validate(dqn,
                            dataset,
                            scoring={'td_error': td_error_scorer},
                            fit_params={'n_epochs': 1})

GridSearchCV
------------

You can also perform grid search to find good hyperparameters.

.. code-block:: python

    from d3rlpy.algos import DQN
    from d3rlpy.datasets import get_cartpole
    from d3rlpy.metrics import td_error_scorer
    from sklearn.model_selection import GridSearchCV

    dataset, env = get_cartpole()

    dqn = DQN()

    gscv = GridSearchCV(estimator=dqn,
                        param_grid={'learning_rate': [1e-4, 3e-4, 1e-3]},
                        scoring={'td_error': td_error_scorer},
                        refit=False)

    gscv.fit(dataset.episodes, n_epochs=1)


parallel execution with multiple GPUs
-------------------------------------

Some scikit-learn utilities provide `n_jobs` option, which enable fitting
process to run in paralell to boost productivity.
Idealy, if you have multiple GPUs, the multiple processes use different GPUs
for computational efficiency.

d3rlpy provides special device assignment mechanism to realize this.

.. code-block:: python

    from d3rlpy.algos import DQN
    from d3rlpy.datasets import get_cartpole
    from d3rlpy.metrics import td_error_scorer
    from d3rlpy.context import parallel
    from sklearn.model_selection import cross_validate

    dataset, env = get_cartpole()

    # enable GPU
    dqn = DQN(use_gpu=True)

    # automatically assign different GPUs for the 4 processes.
    with parallel():
        scores = cross_validate(dqn,
                                dataset,
                                scoring={'td_error': td_error_scorer},
                                fit_params={'n_epochs': 1},
                                n_jobs=4)

If `use_gpu=True` is passed, d3rlpy internally manages GPU device id via
:class:`d3rlpy.gpu.Device` object.
This object is designed for scikit-learn's multi-process implementation that
makes deep copies of the estimator object before dispatching.
The `Device` object will increment its device id when deeply copied under the
paralell context.

.. code-block:: python

    import copy
    from d3rlpy.context import parallel
    from d3rlpy.gpu import Device

    device = Device(0)
    # device.get_id() == 0

    new_device = copy.deepcopy(device)
    # new_device.get_id() == 0

    with parallel():
        new_device = copy.deepcopy(device)
        # new_device.get_id() == 1
        # device.get_id() == 1

        new_device = copy.deepcopy(device)
        # if you have only 2 GPUs, it goes back to 0.
        # new_device.get_id() == 0
        # device.get_id() == 0

    from d3rlpy.algos import DQN

    dqn = DQN(use_gpu=Device(0)) # assign id=0
    dqn = DQN(use_gpu=Device(1)) # assign id=1
