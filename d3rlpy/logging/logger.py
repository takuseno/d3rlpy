import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator, Optional

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
    def get_torch_modules(self) -> dict[str, nn.Module]: ...
    def get_gradients(self) -> Iterator[tuple[str, Float32NDArray]]: ...


class ImplProtocol(Protocol):
    @property
    def modules(self) -> ModuleProtocol:
        raise NotImplementedError


class AlgProtocol(Protocol):
    @property
    def impl(self) -> Optional[ImplProtocol]:
        raise NotImplementedError


class LoggerAdapter(Protocol):
    r"""Interface of LoggerAdapter."""

    def write_params(self, params: dict[str, Any]) -> None:
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
        self,
        epoch: int,
        step: int,
    ) -> None:
        r"""Watch model parameters / gradients during training.

        Args:
            epoch: Epoch.
            step: Training step.
        """


class LoggerAdapterFactory(Protocol):
    r"""Interface of LoggerAdapterFactory."""

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> LoggerAdapter:
        r"""Creates LoggerAdapter.

        This method instantiates ``LoggerAdapter`` with a given
        ``experiment_name``.
        This method is usually called at the beginning of training.

        Args:
            algo: Algorithm.
            experiment_name: Experiment name.
            steps_per_epoch: Number of steps per epoch.
        """
        raise NotImplementedError


class D3RLPyLogger:
    _algo: AlgProtocol
    _adapter: LoggerAdapter
    _experiment_name: str
    _metrics_buffer: defaultdict[str, list[float]]

    def __init__(
        self,
        algo: AlgProtocol,
        adapter_factory: LoggerAdapterFactory,
        experiment_name: str,
        n_steps_per_epoch: int,
        with_timestamp: bool = True,
    ):
        if with_timestamp:
            date = datetime.now().strftime("%Y%m%d%H%M%S")
            self._experiment_name = experiment_name + "_" + date
        else:
            self._experiment_name = experiment_name
        self._algo = algo
        self._adapter = adapter_factory.create(
            algo, self._experiment_name, n_steps_per_epoch
        )
        self._metrics_buffer = defaultdict(list)

    def add_params(self, params: dict[str, Any]) -> None:
        self._adapter.write_params(params)
        LOG.info("Parameters", params=params)

    def add_metric(self, name: str, value: float) -> None:
        self._metrics_buffer[name].append(value)

    def commit(self, epoch: int, step: int) -> dict[str, float]:
        self._adapter.before_write_metric(epoch, step)

        metrics = {}
        for name, buffer in self._metrics_buffer.items():
            metric = sum(buffer) / len(buffer)
            self._adapter.write_metric(epoch, step, name, metric)
            metrics[name] = metric

        LOG.info(
            f"{self._experiment_name}: epoch={epoch} step={step}",
            epoch=epoch,
            step=step,
            metrics=metrics,
        )

        self._adapter.after_write_metric(epoch, step)

        # save model parameter metrics
        self._adapter.watch_model(epoch, step)

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
