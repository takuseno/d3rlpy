from typing import Sequence

import pytest
import torch

from d3rlpy.models.torch.parameters import Parameter


@pytest.mark.parametrize("shape", [(100,)])
def test_parameter(shape: Sequence[int]) -> None:
    data = torch.rand(shape)
    parameter = Parameter(data)

    assert parameter().shape == shape
    assert torch.all(parameter() == data)
