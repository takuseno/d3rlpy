import os
from typing import Any, Dict

import numpy as np

from .logger import (
    AlgProtocol,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
)

__all__ = ["TensorboardAdapter", "TensorboardAdapterFactory"]


class TensorboardAdapter(LoggerAdapter):
    r"""TensorboardAdapter class.

    This class saves metrics for Tensorboard visualization, powered by
    tensorboardX.

    Note that this class does not save models. If you want to save models
    during training, consider ``FileAdapter`` as well.

    Args:
        root_dir (str): Top-level log directory.
        experiment_name (str): Experiment name.
    """

    _algo: AlgProtocol
    _experiment_name: str
    _params: Dict[str, Any]
    _metrics: Dict[str, float]

    def __init__(self, algo: AlgProtocol, root_dir: str, experiment_name: str):
        try:
            from tensorboardX import SummaryWriter
        except ImportError as e:
            raise ImportError("Please install tensorboardX") from e

        self._algo = algo
        self._experiment_name = experiment_name
        logdir = os.path.join(root_dir, "runs", experiment_name)
        self._writer = SummaryWriter(logdir=logdir)
        self._metrics = {}

    def write_params(self, params: Dict[str, Any]) -> None:
        # remove non-scaler values for HParams
        self._params = {k: v for k, v in params.items() if np.isscalar(v)}

    def before_write_metric(self, epoch: int, step: int) -> None:
        self._metrics = {}

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        self._writer.add_scalar(f"metrics/{name}", value, epoch)
        self._metrics[name] = value

    def after_write_metric(self, epoch: int, step: int) -> None:
        self._writer.add_hparams(
            self._params,
            self._metrics,
            name=self._experiment_name,
            global_step=epoch,
        )

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        pass

    def close(self) -> None:
        self._writer.close()

    def watch_model(
        self,
        epoch: int,
        step: int,
    ) -> None:
        assert self._algo.impl
        for name, grad in self._algo.impl.modules.get_gradients():
            self._writer.add_histogram(f"histograms/{name}_grad", grad, epoch)


class TensorboardAdapterFactory(LoggerAdapterFactory):
    r"""TensorboardAdapterFactory class.

    This class instantiates ``TensorboardAdapter`` object.

    Args:
        root_dir (str): Top-level log directory.
    """

    _root_dir: str

    def __init__(self, root_dir: str = "tensorboard_logs"):
        self._root_dir = root_dir

    def create(
        self, algo: AlgProtocol, experiment_name: str, n_steps_per_epoch: int
    ) -> TensorboardAdapter:
        return TensorboardAdapter(algo, self._root_dir, experiment_name)
