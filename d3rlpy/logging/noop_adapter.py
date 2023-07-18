from typing import Any, Dict

from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol

__all__ = ["NoopAdapter", "NoopAdapterFactory"]


class NoopAdapter(LoggerAdapter):
    r"""NoopAdapter class.

    This class does not save anything. This can be used especially when programs
    are not allowed to write things to disks.
    """

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
    r"""NoopAdapterFactory class.

    This class instantiates ``NoopAdapter`` object.
    """

    def create(self, experiment_name: str) -> NoopAdapter:
        return NoopAdapter()
