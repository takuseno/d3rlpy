# pylint: disable=unidiomatic-typecheck
import io
import os

from d3rlpy.base import (
    ImplBase,
    LearnableBase,
    LearnableConfig,
    load_learnable,
    save_config,
)
from d3rlpy.logging import D3RLPyLogger, FileAdapterFactory
from d3rlpy.logging.file_adapter import FileAdapter
from d3rlpy.types import Shape


def _check_reconst_algo(
    algo: LearnableBase[ImplBase, LearnableConfig],
    new_algo: LearnableBase[ImplBase, LearnableConfig],
) -> None:
    assert new_algo.impl is not None
    assert type(new_algo) == type(algo)
    assert algo.observation_shape is not None
    assert new_algo.observation_shape is not None
    if isinstance(algo.observation_shape[0], int):
        assert tuple(new_algo.observation_shape) == algo.observation_shape
    else:
        for new_algo_shape, algo_shape in zip(
            new_algo.observation_shape, algo.observation_shape
        ):
            assert tuple(new_algo_shape) == algo_shape  # type: ignore
    assert new_algo.impl.action_size == algo.action_size

    # check observation scaler
    if algo.observation_scaler is None:
        assert new_algo.observation_scaler is None
    else:
        assert type(algo.observation_scaler) == type(
            new_algo.observation_scaler
        )

    # check action scaler
    if algo.action_scaler is None:
        assert new_algo.action_scaler is None
    else:
        assert type(algo.action_scaler) == type(new_algo.action_scaler)

    # check reward scaler
    if algo.reward_scaler is None:
        assert new_algo.reward_scaler is None
    else:
        assert type(algo.reward_scaler) == type(new_algo.reward_scaler)


def from_json_tester(
    algo: LearnableBase[ImplBase, LearnableConfig],
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    # save params.json
    adapter_factory = FileAdapterFactory("test_data")
    logger = D3RLPyLogger(
        algo=algo,
        adapter_factory=adapter_factory,
        n_steps_per_epoch=1,
        experiment_name="test",
    )
    # save parameters to test_data/test/params.json
    save_config(algo, logger)
    # load params.json
    adapter = logger.adapter
    assert isinstance(adapter, FileAdapter)
    json_path = os.path.join(adapter.logdir, "params.json")
    new_algo = algo.__class__.from_json(json_path)

    _check_reconst_algo(algo, new_algo)


def load_learnable_tester(
    algo: LearnableBase[ImplBase, LearnableConfig],
    observation_shape: Shape,
    action_size: int,
) -> None:
    algo.create_impl(observation_shape, action_size)
    # save as d3
    path = os.path.join("test_data", "algo.d3")
    algo.save(path)
    # load from d3
    new_algo = load_learnable(path)

    assert algo.impl
    assert new_algo.impl
    _check_reconst_algo(algo, new_algo)

    # check weights
    algo_bytes = io.BytesIO()
    algo.impl.save_model(algo_bytes)
    new_algo_bytes = io.BytesIO()
    new_algo.impl.save_model(new_algo_bytes)
    assert algo_bytes.getvalue() == new_algo_bytes.getvalue()
