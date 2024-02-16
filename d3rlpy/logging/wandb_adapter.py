import wandb
from typing import Any, Dict, Optional
from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol


class LoggerWanDBAdapter(LoggerAdapter):

    def __init__(self, project: Optional[str] = None, experiment_name: Optional[str] = None):
        self.run = wandb.init(project=project, name=experiment_name)

    def write_params(self, params: Dict[str, Any]) -> None:
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(self, epoch: int, step: int, name: str, value: float) -> None:
        self.run.log({name: value, 'epoch': epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        # Implement saving model to wandb if needed
        pass

    def close(self) -> None:
        self.run.finish()


class WanDBAdapterFactory(LoggerAdapterFactory):

    _project: str

    def __init__(self, project: Optional[str] = None) -> None:
        super().__init__()
        self._project = project

    def create(self, experiment_name: str) -> LoggerAdapter:
        return LoggerWanDBAdapter(project=self._project, experiment_name=experiment_name)
