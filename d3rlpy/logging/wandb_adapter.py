from typing import Any, Dict, Optional

from .logger import LoggerAdapter, LoggerAdapterFactory, SaveProtocol

__all__ = ["LoggerWanDBAdapter", "WanDBAdapterFactory"]


class LoggerWanDBAdapter(LoggerAdapter):
    r"""WandB Logger Adapter class.

    This class logs data to Weights & Biases (WandB) for experiment tracking.

    Args:
        experiment_name (str): Name of the experiment.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Please install wandb") from e
        self.run = wandb.init(project=project, name=experiment_name)

    def write_params(self, params: Dict[str, Any]) -> None:
        """Writes hyperparameters to WandB config."""
        self.run.config.update(params)

    def before_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed before writing metric."""
        pass

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        """Writes metric to WandB."""
        self.run.log({name: value, "epoch": epoch}, step=step)

    def after_write_metric(self, epoch: int, step: int) -> None:
        """Callback executed after writing metric."""
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        """Saves models to Weights & Biases.

        Not implemented for WandB.
        """
        # Implement saving model to wandb if needed
        pass

    def close(self) -> None:
        """Closes the logger and finishes the WandB run."""
        self.run.finish()


class WanDBAdapterFactory(LoggerAdapterFactory):
    r"""WandB Logger Adapter Factory class.

    This class creates instances of the WandB Logger Adapter for experiment
    tracking.
    """

    _project: str

    def __init__(self, project: Optional[str] = None) -> None:
        """Initialize the WandB Logger Adapter Factory.

        Args:
            project (Optional[str], optional): The name of the WandB project. Defaults to None.
        """
        super().__init__()
        self._project = project

    def create(self, experiment_name: str) -> LoggerAdapter:
        """Creates a WandB Logger Adapter instance.

        Args:
            experiment_name (str): Name of the experiment.

        Returns:
            LoggerAdapter: Instance of the WandB Logger Adapter.
        """
        return LoggerWanDBAdapter(
            project=self._project, experiment_name=experiment_name
        )
