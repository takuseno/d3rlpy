from typing import Any, Dict, List

import pytest
from torch import nn

from d3rlpy.logging import D3RLPyLogger
from d3rlpy.logging.logger import SaveProtocol, TorchModuleProtocol


class StubLoggerAdapter:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.is_write_params_called = False
        self.is_before_write_metric_called = False
        self.is_write_metric_called = False
        self.is_after_write_metric_called = False
        self.is_save_model_called = False
        self.is_close_called = False
        self.is_watch_model_called = False

    def write_params(self, params: Dict[str, Any]) -> None:
        self.is_write_params_called = True

    def before_write_metric(self, epoch: int, step: int) -> None:
        self.is_before_write_metric_called = True

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        assert self.is_before_write_metric_called
        self.is_write_metric_called = True

    def after_write_metric(self, epoch: int, step: int) -> None:
        assert self.is_before_write_metric_called
        assert self.is_write_metric_called
        self.is_after_write_metric_called = True

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        self.is_save_model_called = True

    def close(self) -> None:
        self.is_close_called = True

    def watch_model(
        self,
        epoch: int,
        step: int,
        logging_step: int,
        algo: TorchModuleProtocol,
    ) -> None:
        self.is_watch_model_called = True


class StubLoggerAdapterFactory:
    def create(self, experiment_name: str) -> StubLoggerAdapter:
        return StubLoggerAdapter(experiment_name)


class StubModules:
    def get_torch_modules(self) -> List[nn.Module]:
        return []


class StubImpl:
    modules: StubModules


class StubAlgo:
    impl: StubImpl

    def save(self, fname: str) -> None:
        pass


@pytest.mark.parametrize("with_timestamp", [False, True])
def test_d3rlpy_logger(with_timestamp: bool) -> None:
    logger = D3RLPyLogger(StubLoggerAdapterFactory(), "test", with_timestamp)

    # check experiment_name
    adapter = logger.adapter
    assert isinstance(adapter, StubLoggerAdapter)
    if with_timestamp:
        assert adapter.experiment_name != "test"
    else:
        assert adapter.experiment_name == "test"

    assert not adapter.is_write_params_called
    logger.add_params({"test": 1})
    assert adapter.is_write_params_called

    logger.add_metric("test", 1)
    with logger.measure_time("test"):
        pass

    assert not adapter.is_before_write_metric_called
    assert not adapter.is_write_metric_called
    assert not adapter.is_after_write_metric_called
    metrics = logger.commit(1, 1)
    assert "test" in metrics
    assert "time_test" in metrics
    assert adapter.is_before_write_metric_called
    assert adapter.is_write_metric_called
    assert adapter.is_after_write_metric_called

    assert not adapter.is_save_model_called
    logger.save_model(1, StubAlgo())
    assert adapter.is_save_model_called

    assert not adapter.is_close_called
    logger.close()
    assert adapter.is_close_called

    assert not adapter.is_watch_model_called
    logger.watch_model(1, 1, 1, StubAlgo())
