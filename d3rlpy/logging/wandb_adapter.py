from typing import Any, Dict, Optional

from .logger import (
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
)

__all__ = ["WanDBAdapter", "WanDBAdapterFactory"]


class WanDBAdapter(LoggerAdapter):
    r"""WandB Logger Adapter class.

    This class logs data to Weights & Biases (WandB) for experiment tracking.

    Args:
        algo: Algorithm.
        experiment_name (str): Name of the experiment.
        n_steps_per_epoch: Number of steps per epoch.
        project: Project name.
    """

    def __init__(
        self,
        algo: AlgProtocol,
        experiment_name: str,
        n_steps_per_epoch: int,
        project: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        assert algo.impl
        self.run = wandb.init(project=project, name=experiment_name)
        self.run.watch(
            tuple(algo.impl.modules.get_torch_modules().values()),
            log="gradients",
            log_freq=n_steps_per_epoch,
        )
        self._is_model_watched = False

    def write_params(self, params: Dict[str, Any]) -> None:
        """Writes hyperparameters to WandB config."""
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed before writing metric."""

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        """Writes metric to WandB."""
        self.run.log({name: value, "epoch": epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed after writing metric."""

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        """Saves models to Weights & Biases.

        Not implemented for WandB.
        """
        # Implement saving model to wandb if needed

    def close(self) -> None:
        """Closes the logger and finishes the WandB run."""
        self.run.finish()

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        pass


class WanDBAdapterFactory(LoggerAdapterFactory):
    r"""WandB Logger Adapter Factory class.

    This class creates instances of the WandB Logger Adapter for experiment
    tracking.

    Args:
        project (Optional[str], optional): The name of the WandB project. Defaults to None.
    """

    _project: Optional[str]

    def __init__(self, project: Optional[str] = None) -> None:
        self._project = project

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> LoggerAdapter:
        return WanDBAdapter(
            algo=algo,
            experiment_name=experiment_name,
            n_steps_per_epoch=n_steps_per_epoch,
            project=self._project,
        )
