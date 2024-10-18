from typing import Sequence

import numpy as np
import pytest
import torch

from d3rlpy.constants import PositionEncodingType
from d3rlpy.models.builders import (
    create_categorical_policy,
    create_continuous_decision_transformer,
    create_continuous_q_function,
    create_deterministic_policy,
    create_deterministic_residual_policy,
    create_discrete_decision_transformer,
    create_discrete_q_function,
    create_normal_policy,
    create_parameter,
    create_vae_decoder,
    create_vae_encoder,
    create_value_function,
)
from d3rlpy.models.encoders import DefaultEncoderFactory, EncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.models.torch import (
    ContinuousEnsembleQFunctionForwarder,
    DiscreteEnsembleQFunctionForwarder,
    get_parameter,
)
from d3rlpy.models.torch.imitators import VAEDecoder, VAEEncoder
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
        observation_shape,
        action_size,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
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
        observation_shape,
        action_size,
        scale,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
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
        observation_shape,
        action_size,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
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
        observation_shape,
        action_size,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
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

    q_funcs, forwarder = create_discrete_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        device="cpu:0",
        enable_ddp=False,
        n_ensembles=n_ensembles,
    )

    assert isinstance(forwarder, DiscreteEnsembleQFunctionForwarder)

    # check share_encoder
    encoder = q_funcs[0].encoder
    for q_func in list(q_funcs.modules())[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, *observation_shape))
    y = forwarder.compute_expected_q(x)
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

    q_funcs, forwarder = create_continuous_q_function(
        observation_shape,
        action_size,
        encoder_factory,
        q_func_factory,
        device="cpu:0",
        enable_ddp=False,
        n_ensembles=n_ensembles,
    )

    assert isinstance(forwarder, ContinuousEnsembleQFunctionForwarder)

    # check share_encoder
    encoder = q_funcs[0].encoder
    for q_func in list(q_funcs.modules())[1:]:
        if share_encoder:
            assert encoder is q_func.encoder
        else:
            assert encoder is not q_func.encoder

    x = torch.rand((batch_size, *observation_shape))
    action = torch.rand(batch_size, action_size)
    y = forwarder.compute_expected_q(x, action)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_vae_encoder(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    vae_encoder = create_vae_encoder(
        observation_shape,
        action_size,
        latent_size,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
    )

    assert isinstance(vae_encoder, VAEEncoder)

    x = torch.rand((batch_size, *observation_shape))
    action = torch.rand(batch_size, action_size)
    dist = vae_encoder(x, action)
    assert dist.mean.shape == (batch_size, latent_size)


@pytest.mark.parametrize("observation_shape", [(4, 84, 84), (100,)])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("latent_size", [32])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
def test_create_vae_decoder(
    observation_shape: Sequence[int],
    action_size: int,
    latent_size: int,
    batch_size: int,
    encoder_factory: EncoderFactory,
) -> None:
    vae_decoder = create_vae_decoder(
        observation_shape,
        action_size,
        latent_size,
        encoder_factory,
        device="cpu:0",
        enable_ddp=False,
    )

    assert isinstance(vae_decoder, VAEDecoder)

    x = torch.rand((batch_size, *observation_shape))
    latent = torch.rand(batch_size, latent_size)
    y = vae_decoder(x, latent)
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
        observation_shape, encoder_factory, device="cpu:0", enable_ddp=False
    )

    assert isinstance(v_func, ValueFunction)

    x = torch.rand((batch_size, *observation_shape))
    y = v_func(x)
    assert y.shape == (batch_size, 1)


@pytest.mark.parametrize("shape", [(100,)])
def test_create_parameter(shape: Sequence[int]) -> None:
    x = np.random.random()
    parameter = create_parameter(shape, x, device="cpu:0", enable_ddp=False)

    assert len(list(parameter.parameters())) == 1
    assert np.allclose(get_parameter(parameter).detach().numpy(), x)


@pytest.mark.parametrize("observation_shape", [(100,), (4, 84, 84)])
@pytest.mark.parametrize("encoder_factory", [DefaultEncoderFactory()])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("num_heads", [2])
@pytest.mark.parametrize("max_timestep", [10])
@pytest.mark.parametrize("num_layers", [3])
@pytest.mark.parametrize("context_size", [10])
@pytest.mark.parametrize("dropout", [0.1])
@pytest.mark.parametrize("activation_type", ["relu"])
@pytest.mark.parametrize(
    "position_encoding_type",
    [PositionEncodingType.SIMPLE, PositionEncodingType.GLOBAL],
)
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
    position_encoding_type: PositionEncodingType,
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
        enable_ddp=False,
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
@pytest.mark.parametrize(
    "position_encoding_type",
    [PositionEncodingType.SIMPLE, PositionEncodingType.GLOBAL],
)
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
    position_encoding_type: PositionEncodingType,
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
        embed_activation_type=activation_type,
        position_encoding_type=position_encoding_type,
        device="cpu:0",
        enable_ddp=False,
    )

    assert isinstance(transformer, DiscreteDecisionTransformer)

    x = torch.rand(batch_size, context_size, *observation_shape)
    action = torch.randint(0, action_size, size=(batch_size, context_size))
    rtg = torch.rand(batch_size, context_size, 1)
    timesteps = torch.randint(0, max_timestep, size=(batch_size, context_size))
    probs, logits = transformer(x, action, rtg, timesteps)

    assert probs.shape == (batch_size, context_size, action_size)
    assert logits.shape == (batch_size, context_size, action_size)
