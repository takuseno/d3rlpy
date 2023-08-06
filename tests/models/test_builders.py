from typing import Sequence

import numpy as np
import pytest
import torch

from d3rlpy.models.builders import (
    create_categorical_policy,
    create_conditional_vae,
    create_continuous_decision_transformer,
    create_continuous_q_function,
    create_deterministic_policy,
    create_deterministic_regressor,
    create_deterministic_residual_policy,
    create_discrete_decision_transformer,
    create_discrete_imitator,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
    create_probablistic_regressor,
    create_value_function,
)
from d3rlpy.models.encoders import DefaultEncoderFactory, EncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.torch import (
    EnsembleContinuousQFunction,
    EnsembleDiscreteQFunction,
)
from d3rlpy.models.torch.imitators import (
    ConditionalVAE,
    DeterministicRegressor,
    DiscreteImitator,
    ProbablisticRegressor,
)
from d3rlpy.models.torch.policies import (
    CategoricalPolicy,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    NormalPolicy,
)
from d3rlpy.models.torch.transformers import (
    ContinuousDecisionTransformer,
    DiscreteDecisionTransformer,
)
from d3rlpy.models.torch.v_functions import ValueFunction


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_policy(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    policy = create_deterministic_policy(
        observation_shape, action_size, encoder_factory, device="cpu:0"
    )

    assert isinstance(policy, DeterministicPolicy)

    x = torch.rand((batch_size, *observation_shape))
    y = policy(x)
    assert y.mu.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("scale", [0.05])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_residual_policy(
    observation_shape: Sequence[int],
    action_size: int,
    scale: float,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    policy = create_deterministic_residual_policy(
        observation_shape, action_size, scale, encoder_factory, device="cpu:0"
    )

    assert isinstance(policy, DeterministicResidualPolicy)

    x = torch.rand((batch_size, *observation_shape))
    action = torch.rand(batch_size, action_size)
    y = policy(x, action)
    assert y.mu.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_normal_policy(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    policy = create_normal_policy(
        observation_shape, action_size, encoder_factory, device="cpu:0"
    )

    assert isinstance(policy, NormalPolicy)

    x = torch.rand((batch_size, *observation_shape))
    y = policy(x)
    assert y.mu.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_categorical_policy(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    policy = create_categorical_policy(
        observation_shape, action_size, encoder_factory, device="cpu:0"
    )

    assert isinstance(policy, CategoricalPolicy)

    x = torch.rand((batch_size, *observation_shape))
    dist = policy(x)
    assert dist.probs.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 5])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_create_discrete_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    n_ensembles: int,
    encoder_factory: EncoderFactory,
    share_encoder: bool,
) -> None:
    q_func_factory = MeanQFunctionFactory(share_encoder=share_encoder)

    q_func = create_discrete_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        device="cpu:0",
        n_ensembles=n_ensembles,
    )

    assert isinstance(q_func, EnsembleDiscreteQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, *observation_shape))
    y = q_func(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("n_ensembles", [1, 2])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("share_encoder", [False, True])
def test_create_continuous_q_function(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    n_ensembles: int,
    encoder_factory: EncoderFactory,
    share_encoder: bool,
) -> None:
    q_func_factory = MeanQFunctionFactory(share_encoder=share_encoder)

    q_func = create_continuous_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        device="cpu:0",
        n_ensembles=n_ensembles,
    )

    assert isinstance(q_func, EnsembleContinuousQFunction)

    # check share_encoder
    encoder = q_func.q_funcs[0].encoder
    for q_func in q_func.q_funcs[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, *observation_shape))
    action = torch.rand(batch_size, action_size)
    y = q_func(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_conditional_vae(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    vae = create_conditional_vae(
        observation_shape,
        action_size,
        latent_size,
        encoder_factory,
        device="cpu:0",
    )

    assert isinstance(vae, ConditionalVAE)

    x = torch.rand((batch_size, *observation_shape))
    action = torch.rand(batch_size, action_size)
    y = vae(x, action)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("beta", [1e-2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_discrete_imitator(
    observation_shape: Sequence[int],
    action_size: int,
    beta: float,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    imitator = create_discrete_imitator(
        observation_shape, action_size, beta, encoder_factory, device="cpu:0"
    )

    assert isinstance(imitator, DiscreteImitator)

    x = torch.rand((batch_size, *observation_shape))
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_deterministic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    imitator = create_deterministic_regressor(
        observation_shape, action_size, encoder_factory, device="cpu:0"
    )

    assert isinstance(imitator, DeterministicRegressor)

    x = torch.rand((batch_size, *observation_shape))
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_probablistic_regressor(
    observation_shape: Sequence[int],
    action_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    imitator = create_probablistic_regressor(
        observation_shape, action_size, encoder_factory, device="cpu:0"
    )

    assert isinstance(imitator, ProbablisticRegressor)

    x = torch.rand((batch_size, *observation_shape))
    y = imitator(x)
    assert y.shape == (batch_size, action_size)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("batch_size", [32])
def test_create_value_function(
    observation_shape: Sequence[int],
    encoder_factory: EncoderFactory,
    batch_size: int,
) -> None:
    v_func = create_value_function(
        observation_shape, encoder_factory, device="cpu:0"
    )

    assert isinstance(v_func, ValueFunction)

    x = torch.rand((batch_size, *observation_shape))
    y = v_func(x)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("shape", [(100,)])
def test_create_parameter(shape: Sequence[int]) -> None:
    x = np.random.random()
    parameter = create_parameter(shape, x, device="cpu:0")

    assert len(list(parameter.parameters())) == 1
    assert np.allclose(parameter().detach().numpy(), x)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("max_timestep", [10])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.1])
@pytest.mark.parametrize("activation_type", ["relu"])
@pytest.mark.parametrize("position_encoding_type", ["simple"])
@pytest.mark.parametrize("batch_size", [32])
def test_create_continuous_decision_transformer(
    observation_shape: Sequence[int],
    encoder_factory: EncoderFactory,
    action_size: int,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    dropout: float,
    activation_type: str,
    position_encoding_type: str,
    batch_size: int,
) -> None:
    transformer = create_continuous_decision_transformer(
        observation_shape=observation_shape,
        action_size=action_size,
        encoder_factory=encoder_factory,
        num_heads=num_heads,
        max_timestep=max_timestep,
        num_layers=num_layers,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        embed_dropout=dropout,
        activation_type=activation_type,
        position_encoding_type=position_encoding_type,
        device="cpu:0",
    )

    assert isinstance(transformer, ContinuousDecisionTransformer)

    x = torch.rand(batch_size, context_size, *observation_shape)
    action = torch.rand(batch_size, context_size, action_size)
    rtg = torch.rand(batch_size, context_size, 1)
    timesteps = torch.randint(0, max_timestep, size=(batch_size, context_size))
    y = transformer(x, action, rtg, timesteps)

    assert y.shape == (batch_size, context_size, action_size)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("max_timestep", [10])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.1])
@pytest.mark.parametrize("activation_type", ["relu"])
@pytest.mark.parametrize("position_encoding_type", ["simple"])
@pytest.mark.parametrize("batch_size", [32])
def test_create_discrete_decision_transformer(
    observation_shape: Sequence[int],
    encoder_factory: EncoderFactory,
    action_size: int,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    dropout: float,
    activation_type: str,
    position_encoding_type: str,
    batch_size: int,
) -> None:
    transformer = create_discrete_decision_transformer(
        observation_shape=observation_shape,
        action_size=action_size,
        encoder_factory=encoder_factory,
        num_heads=num_heads,
        max_timestep=max_timestep,
        num_layers=num_layers,
        context_size=context_size,
        attn_dropout=dropout,
        resid_dropout=dropout,
        embed_dropout=dropout,
        activation_type=activation_type,
        position_encoding_type=position_encoding_type,
        device="cpu:0",
    )

    assert isinstance(transformer, DiscreteDecisionTransformer)

    x = torch.rand(batch_size, context_size, *observation_shape)
    action = torch.randint(0, action_size, size=(batch_size, context_size))
    rtg = torch.rand(batch_size, context_size, 1)
    timesteps = torch.randint(0, max_timestep, size=(batch_size, context_size))
    probs, logits = transformer(x, action, rtg, timesteps)

    assert probs.shape == (batch_size, context_size, action_size)
    assert logits.shape == (batch_size, context_size, action_size)
