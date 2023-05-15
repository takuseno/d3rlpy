import numpy as np
import torch

from d3rlpy.preprocessing.base import add_leading_dims, add_leading_dims_numpy


def test_add_leading_dims() -> None:
    x = torch.rand(3)
    target = torch.rand(1, 2, 3)
    assert add_leading_dims(x, target).shape == (1, 1, 3)


def test_add_leading_dims_numpy() -> None:
    x = np.random.random(3)
    target = np.random.random((1, 2, 3))
    assert add_leading_dims_numpy(x, target).shape == (1, 1, 3)
