import pytest

from d3rlpy.dynamics.torch.probabilistic_ensemble_dynamics_impl import (
    ProbabilisticEnsembleDynamicsImpl,
)
from d3rlpy.models.encoders import DefaultEncoderFactory
from d3rlpy.models.optimizers import AdamFactory
from tests.algos.algo_test import (
    DummyActionScaler,
    DummyRewardScaler,
    DummyScaler,
)
from tests.dynamics.dynamics_test import torch_impl_tester


@pytest.mark.parametrize("observation_shape", [(100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("learning_rate", [1e-3])
@pytest.mark.parametrize("optim_factory", [AdamFactory()])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("n_ensembles", [5])
@pytest.mark.parametrize("variance_type", ["max"])
@pytest.mark.parametrize("discrete_action", [False, True])
@pytest.mark.parametrize("scaler", [None, DummyScaler()])
@pytest.mark.parametrize("action_scaler", [None, DummyActionScaler()])
@pytest.mark.parametrize("reward_scaler", [None, DummyRewardScaler()])
def test_probabilistic_ensemble_dynamics_impl(
    observation_shape,
    action_size,
    learning_rate,
    optim_factory,
    encoder_factory,
    n_ensembles,
    variance_type,
    discrete_action,
    scaler,
    action_scaler,
    reward_scaler,
):
    impl = ProbabilisticEnsembleDynamicsImpl(
        observation_shape=observation_shape,
        action_size=action_size,
        learning_rate=learning_rate,
        optim_factory=optim_factory,
        encoder_factory=encoder_factory,
        n_ensembles=n_ensembles,
        variance_type=variance_type,
        discrete_action=discrete_action,
        use_gpu=None,
        scaler=scaler,
        action_scaler=action_scaler if not discrete_action else None,
        reward_scaler=reward_scaler,
    )
    impl.build()
    torch_impl_tester(impl, discrete=discrete_action, n_ensembles=n_ensembles)
