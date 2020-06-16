import torch
import pytest
import copy

from skbrl.algos.torch.utility import soft_sync, hard_sync


@pytest.mark.parametrize('tau', [0.05])
@pytest.mark.parametrize('input_size', [32])
@pytest.mark.parametrize('output_size', [32])
def test_soft_sync(tau, input_size, output_size):
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)
    original = copy.deepcopy(targ_module)

    soft_sync(targ_module, module, tau)

    module_params = module.parameters()
    targ_params = targ_module.parameters()
    original_params = original.parameters()
    for p, targ_p, orig_p in zip(module_params, targ_params, original_params):
        assert torch.allclose(p * tau + orig_p * (1.0 - tau), targ_p)


@pytest.mark.parametrize('input_size', [32])
@pytest.mark.parametrize('output_size', [32])
def test_hard_sync(input_size, output_size):
    module = torch.nn.Linear(input_size, output_size)
    targ_module = torch.nn.Linear(input_size, output_size)

    hard_sync(targ_module, module)

    for p, targ_p in zip(module.parameters(), targ_module.parameters()):
        assert torch.allclose(targ_p, p)
