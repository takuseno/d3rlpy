import json
import os
from enum import Enum, IntEnum
from typing import Any, Dict, Optional

import numpy as np

from .logger import (
    LOG,
    LoggerAdapter,
    LoggerAdapterFactory,
    SaveProtocol,
    TorchModuleProtocol,
)

__all__ = ["FileAdapter", "FileAdapterFactory"]


# default json encoder for numpy objects
def default_json_encoder(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (Enum, IntEnum)):
        return obj.value
    raise ValueError(f"invalid object type: {type(obj)}")


class FileAdapter(LoggerAdapter):
    r"""FileAdapter class.

    This class saves metrics as CSV files, hyperparameters as json file and
    models as d3 files.

    Args:
        logdir (str): Log directory.
    """

    _logdir: str

    def __init__(self, logdir: str):
        self._logdir = logdir
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)
            LOG.info(f"Directory is created at {self._logdir}")

    def write_params(self, params: Dict[str, Any]) -> None:
        # save dictionary as json file
        params_path = os.path.join(self._logdir, "params.json")
        with open(params_path, "w") as f:
            json_str = json.dumps(
                params, default=default_json_encoder, indent=2
            )
            f.write(json_str)

    def before_write_metric(self, epoch: int, step: int) -> None:
        pass

    def write_metric(
        self, epoch: int, step: int, name: str, value: float
    ) -> None:
        path = os.path.join(self._logdir, f"{name}.csv")
        with open(path, "a") as f:
            print(f"{epoch},{step},{value}", file=f)

    def after_write_metric(self, epoch: int, step: int) -> None:
        pass

    def save_model(self, epoch: int, algo: SaveProtocol) -> None:
        # save entire model
        model_path = os.path.join(self._logdir, f"model_{epoch}.d3")
        algo.save(model_path)
        LOG.info(f"Model parameters are saved to {model_path}")

    def close(self) -> None:
        pass

    @property
    def logdir(self) -> str:
        return self._logdir

    def watch_model(
        self,
        epoch: int,
        step: int,
        logging_steps: Optional[int],
        algo: TorchModuleProtocol,
    ) -> None:
        if logging_steps is not None and step % logging_steps == 0:
            for name, grad in algo.impl.modules.get_gradients():
                path = os.path.join(self._logdir, f"{name}_grad.csv")
                with open(path, "a") as f:
                    min_grad = grad.min()
                    max_grad = grad.max()
                    mean_grad = grad.mean()
                    print(
                        f"{epoch},{step},{name},{min_grad},{max_grad},{mean_grad}",
                        file=f,
                    )


class FileAdapterFactory(LoggerAdapterFactory):
    r"""FileAdapterFactory class.

    This class instantiates ``FileAdapter`` object.
    Log directory will be created at ``<root_dir>/<experiment_name>``.

    Args:
        root_dir (str): Top-level log directory.
    """

    _root_dir: str

    def __init__(self, root_dir: str = "d3rlpy_logs"):
        self._root_dir = root_dir

    def create(self, experiment_name: str) -> FileAdapter:
        logdir = os.path.join(self._root_dir, experiment_name)
        return FileAdapter(logdir)
