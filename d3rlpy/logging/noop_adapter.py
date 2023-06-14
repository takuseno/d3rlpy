from typing import Any, Dict

from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol

__all__ = ["NoopAdapter", "NoopAdapterFactory"]


class NoopAdapter(LoggerAdapter):
    def write_params(self, params: Dict[str, Any]) -> None:
        pass

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        pass

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        pass

    def close(self) -> None:
        pass


class NoopAdapterFactory(LoggerAdapterFactory):
    def create(self, experiment_name: str) -> NoopAdapter:
        return NoopAdapter()
