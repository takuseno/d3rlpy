import random

import numpy as np
import torch

from ._version import __version__
from . import algos
from . import augmentation
from . import dynamics
from . import envs
from . import metrics
from . import models
from . import online
from . import ope
from . import preprocessing
from . import wrappers
from . import dataset
from . import datasets


def seed(n: int) -> None:
    """Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True
