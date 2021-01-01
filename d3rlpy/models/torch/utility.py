import torch.nn as nn

from typing import Sequence, cast
from d3rlpy.encoders import EncoderFactory
from d3rlpy.q_functions import QFunctionFactory
from .encoders import Encoder, EncoderWithAction
from .q_functions import EnsembleDiscreteQFunction, EnsembleContinuousQFunction
from .policies import DeterministicPolicy, DeterministicResidualPolicy
from .policies import NormalPolicy, CategoricalPolicy
from .imitators import ConditionalVAE, DiscreteImitator
from .imitators import DeterministicRegressor, ProbablisticRegressor
from .v_functions import ValueFunction
from .dynamics import EnsembleDynamics, ProbablisticDynamics


def create_discrete_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
    bootstrap: bool = False,
    share_encoder: bool = False,
) -> EnsembleDiscreteQFunction:
    if share_encoder:
        encoder = encoder_factory.create(observation_shape)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not share_encoder:
            encoder = encoder_factory.create(observation_shape)
        q_funcs.append(q_func_factory.create(encoder, action_size))
    return EnsembleDiscreteQFunction(q_funcs, bootstrap)


def create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    n_ensembles: int = 1,
    bootstrap: bool = False,
    share_encoder: bool = False,
) -> EnsembleContinuousQFunction:
    if share_encoder:
        encoder = encoder_factory.create(observation_shape, action_size)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    for _ in range(n_ensembles):
        if not share_encoder:
            encoder = encoder_factory.create(observation_shape, action_size)
        q_funcs.append(q_func_factory.create(encoder))
    return EnsembleContinuousQFunction(q_funcs, bootstrap)


def create_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return DeterministicPolicy(encoder, action_size)


def create_deterministic_residual_policy(
    observation_shape: Sequence[int],
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
) -> DeterministicResidualPolicy:
    encoder = encoder_factory.create(observation_shape, action_size)
    assert isinstance(encoder, EncoderWithAction)
    return DeterministicResidualPolicy(encoder, scale)


def create_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> NormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return NormalPolicy(
        encoder,
        action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )


def create_categorical_policy(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return CategoricalPolicy(encoder, action_size)


def create_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    beta: int,
    encoder_factory: EncoderFactory,
) -> ConditionalVAE:
    encoder_encoder = encoder_factory.create(observation_shape, action_size)
    decoder_encoder = encoder_factory.create(observation_shape, latent_size)
    assert isinstance(encoder_encoder, EncoderWithAction)
    assert isinstance(decoder_encoder, EncoderWithAction)
    return ConditionalVAE(encoder_encoder, decoder_encoder, beta)


def create_discrete_imitator(
    observation_shape: Sequence[int],
    action_size: int,
    beta: float,
    encoder_factory: EncoderFactory,
) -> DiscreteImitator:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return DiscreteImitator(encoder, action_size, beta)


def create_deterministic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> DeterministicRegressor:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return DeterministicRegressor(encoder, action_size)


def create_probablistic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
) -> ProbablisticRegressor:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return ProbablisticRegressor(encoder, action_size)


def create_value_function(
    observation_shape: Sequence[int], encoder_factory: EncoderFactory
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    assert isinstance(encoder, Encoder)
    return ValueFunction(encoder)


def create_probablistic_dynamics(
    observation_shape: Sequence[int],
    action_size: int,
    encoder_factory: EncoderFactory,
    n_ensembles: int = 5,
    discrete_action: bool = False,
) -> EnsembleDynamics:
    models = []
    for _ in range(n_ensembles):
        encoder = encoder_factory.create(
            observation_shape=observation_shape,
            action_size=action_size,
            discrete_action=discrete_action,
        )
        assert isinstance(encoder, EncoderWithAction)
        model = ProbablisticDynamics(encoder)
        models.append(model)
    return EnsembleDynamics(models)
