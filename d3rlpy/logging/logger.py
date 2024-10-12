import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, DefaultDict, Dict, Iterator, List

import numpy as np
import structlog
from torch import nn
from typing_extensions import Protocol

from ..types import Float32NDArray

__all__ = [
    "LOG",
    "set_log_context",
    "D3RLPyLogger",
    "LoggerAdapter",
    "LoggerAdapterFactory",
]

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M.%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
)


LOG: structlog.BoundLogger = structlog.get_logger(__name__)


def set_log_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


class SaveProtocol(Protocol):
    def save(self, fname: str) -> None: ...


class ModuleProtocol(Protocol):
    def get_torch_modules(self) -> List[nn.Module]: ...


class ImplProtocol(Protocol):
    modules: ModuleProtocol


class TorchModuleProtocol(Protocol):
    impl: ImplProtocol


class LoggerAdapter(Protocol):
    r"""Interface of LoggerAdapter."""

    def write_params(self, params: Dict[str, Any]) -> None:
        r"""Writes hyperparameters.

        Args:
            params: Dictionary of hyperparameters.
        """

    def before_write_metric(self, epoch: int, step: int) -> None:
        r"""Callback executed before write_metric method.

        Args:
            epoch: Epoch.
            step: Training step.
        """

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        r"""Writes metric.

        Args:
            epoch: Epoch.
            step: Training step.
            name: Metric name.
            value: Metric value.
        """

    def write_histogram(
        self, epoch: int, step: int, name: str, values: Float32NDArray
    ) -> None:
        r"""Writes histogram.

        # TODO
        Args:
            epoch:
            step:
            name:
            values:
        """

    def after_write_metric(self, epoch: int, step: int) -> None:
        r"""Callback executed after write_metric method.

        Args:
            epoch: Epoch.
            step: Training step.
        """

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        r"""Saves models.

        Args:
            epoch: Epoch.
            algo: Algorithm that provides ``save`` method.
        """

    def close(self) -> None:
        r"""Closes this LoggerAdapter."""

    def watch_model(
        self, logging_steps: int, algo: TorchModuleProtocol
    ) -> None:
        r"""TODO: Docstring and type"""


class LoggerAdapterFactory(Protocol):
    r"""Interface of LoggerAdapterFactory."""

    def create(self, experiment_name: str) -> LoggerAdapter:
        r"""Creates LoggerAdapter.

        This method instantiates ``LoggerAdapter`` with a given
        ``experiment_name``.
        This method is usually called at the beginning of training.

        Args:
            experiment_name: Experiment name.
        """
        raise NotImplementedError


class D3RLPyLogger:
    _adapter: LoggerAdapter
    _experiment_name: str
    _metrics_buffer: DefaultDict[str, List[float]]

    def __init__(
        self,
        adapter_factory: LoggerAdapterFactory,
        experiment_name: str,
        with_timestamp: bool = True,
    ):
        if with_timestamp:
            date = datetime.now().strftime("%Y%m%d%H%M%S")
            self._experiment_name = experiment_name + "_" + date
        else:
            self._experiment_name = experiment_name
        self._adapter = adapter_factory.create(self._experiment_name)
        self._metrics_buffer = defaultdict(list)
        self._histogram_metrics_buffer = defaultdict(list)

    def add_params(self, params: Dict[str, Any]) -> None:
        self._adapter.write_params(params)
        LOG.info("Parameters", params=params)

    def add_metric(self, name: str, value: float) -> None:
        self._metrics_buffer[name].append(value)

    def add_histogram(self, name: str, values: Float32NDArray) -> None:
        self._histogram_metrics_buffer[name].append(values)

    def commit(self, epoch: int, step: int) -> Dict[str, float]:
        self._adapter.before_write_metric(epoch, step)

        metrics = {}
        for name, buffer in self._metrics_buffer.items():
            metric = sum(buffer) / len(buffer)
            self._adapter.write_metric(epoch, step, name, metric)
            metrics[name] = metric

        for name, buffer in self._histogram_metrics_buffer.items():
            histogram_values = np.concatenate(buffer)
            self._adapter.write_histogram(epoch, step, name, histogram_values)

        LOG.info(
            f"{self._experiment_name}: epoch={epoch} step={step}",
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

        self._adapter.after_write_metric(epoch, step)

        # initialize metrics buffer
        self._metrics_buffer.clear()
        return metrics

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        self._adapter.save_model(epoch, algo)

    def close(self) -> None:
        self._adapter.close()

    @contextmanager
    def measure_time(self, name: str) -> Iterator[None]:
        name = "time_" + name
        start = time.time()
        try:
            yield
        finally:
            self.add_metric(name, time.time() - start)

    @property
    def adapter(self) -> LoggerAdapter:
        return self._adapter

    def watch_model(self, logging_steps, algo: TorchModuleProtocol) -> None:
        self._adapter.watch_model(logging_steps, algo)
