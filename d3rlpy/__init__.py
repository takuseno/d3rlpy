import numpy as np
import torch
import random

from ._version import __version__


def seed(n):
    """ Sets random seed value.

    Args:
        n (int): seed value.

    """
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.backends.cudnn.deterministic = True
