import pytest

from d3rlpy.dynamics.torch.mopo_impl import MOPOImpl
from tests.algos.algo_test import DummyScaler
from tests.dynamics.dynamics_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('n_ensembles', [5])
@pytest.mark.parametrize('lam', [1.0])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
def test_mopo_impl(observation_shape, action_size, learning_rate, eps,
                   n_ensembles, lam, use_batch_norm, scaler):
    impl = MOPOImpl(observation_shape,
                    action_size,
                    learning_rate,
                    eps,
                    n_ensembles,
                    lam,
                    use_batch_norm,
                    use_gpu=False,
                    scaler=scaler)
    torch_impl_tester(impl, discrete=False)
