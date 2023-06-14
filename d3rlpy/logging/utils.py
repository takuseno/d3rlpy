from typing import Any, Dict, Sequence

from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol

__all__ = ["CombineAdapter", "CombineAdapterFactory"]


class CombineAdapter(LoggerAdapter):
    def __init__(self, adapters: Sequence[LoggerAdapter]):
        self._adapters = adapters

    def write_params(self, params: Dict[str, Any]) -> None:
        for adapter in self._adapters:
            adapter.write_params(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        for adapter in self._adapters:
            adapter.before_write_metric(epoch, step)

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        for adapter in self._adapters:
            adapter.write_metric(epoch, step, name, value)

    def after_write_metric(self, epoch: int, step: int) -> None:
        for adapter in self._adapters:
            adapter.after_write_metric(epoch, step)

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        for adapter in self._adapters:
            adapter.save_model(epoch, algo)

    def close(self) -> None:
        for adapter in self._adapters:
            adapter.close()


class CombineAdapterFactory(LoggerAdapterFactory):
    _adapter_factories: Sequence[LoggerAdapterFactory]

    def __init__(self, adapter_factories: Sequence[LoggerAdapterFactory]):
        self._adapter_factories = adapter_factories

    def create(self, experiment_name: str) -> CombineAdapter:
        return CombineAdapter(
            [
                factory.create(experiment_name)
                for factory in self._adapter_factories
            ]
        )
