from typing import Sequence, Tuple, cast, Optional

import torch
from torch import nn

from ..constants import PositionEncodingType
from ..torch_utility import wrap_model_by_ddp
from ..types import Shape
from .encoders import EncoderFactory
from .q_functions import QFunctionFactory
from .torch import (
    CategoricalPolicy,
    ContinuousDecisionTransformer,
    ContinuousEnsembleQFunctionForwarder,
    DeterministicPolicy,
    DeterministicResidualPolicy,
    DiscreteDecisionTransformer,
    DiscreteEnsembleQFunctionForwarder,
    GlobalPositionEncoding,
    NormalPolicy,
    Parameter,
    PositionEncoding,
    SimplePositionEncoding,
    VAEDecoder,
    VAEEncoder,
    ValueFunction,
    compute_output_size,
)
from .utility import create_activation

__all__ = [
    "create_discrete_q_function",
    "create_continuous_q_function",
    "create_deterministic_policy",
    "create_deterministic_residual_policy",
    "create_categorical_policy",
    "create_normal_policy",
    "create_vae_encoder",
    "create_vae_decoder",
    "create_value_function",
    "create_parameter",
    "create_continuous_decision_transformer",
    "create_discrete_decision_transformer",
]


def create_discrete_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    enable_ddp: bool,
    n_ensembles: int = 1,
) -> Tuple[nn.ModuleList, DiscreteEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create(observation_shape)
        hidden_size = compute_output_size([observation_shape], encoder)
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create(observation_shape)
            hidden_size = compute_output_size([observation_shape], encoder)
        q_func, forwarder = q_func_factory.create_discrete(
            encoder, hidden_size, action_size
        )
        q_func.to(device)
        if enable_ddp:
            q_func = wrap_model_by_ddp(q_func)
            forwarder.set_q_func(q_func)
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    ensemble_forwarder = DiscreteEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


def create_continuous_q_function(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    q_func_factory: QFunctionFactory,
    device: str,
    enable_ddp: bool,
    n_ensembles: int = 1,
) -> Tuple[nn.ModuleList, ContinuousEnsembleQFunctionForwarder]:
    if q_func_factory.share_encoder:
        encoder = encoder_factory.create_with_action(
            observation_shape, action_size
        )
        hidden_size = compute_output_size(
            [observation_shape, (action_size,)], encoder
        )
        # normalize gradient scale by ensemble size
        for p in cast(nn.Module, encoder).parameters():
            p.register_hook(lambda grad: grad / n_ensembles)

    q_funcs = []
    forwarders = []
    for _ in range(n_ensembles):
        if not q_func_factory.share_encoder:
            encoder = encoder_factory.create_with_action(
                observation_shape, action_size
            )
            hidden_size = compute_output_size(
                [observation_shape, (action_size,)], encoder
            )
        q_func, forwarder = q_func_factory.create_continuous(
            encoder, hidden_size
        )
        q_func.to(device)
        if enable_ddp:
            q_func = wrap_model_by_ddp(q_func)
            forwarder.set_q_func(q_func)
        q_funcs.append(q_func)
        forwarders.append(forwarder)
    q_func_modules = nn.ModuleList(q_funcs)
    ensemble_forwarder = ContinuousEnsembleQFunctionForwarder(
        forwarders, action_size
    )
    return q_func_modules, ensemble_forwarder


def create_deterministic_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
) -> DeterministicPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = DeterministicPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
    )
    policy.to(device)
    if enable_ddp:
        policy = wrap_model_by_ddp(policy)
    return policy


def create_deterministic_residual_policy(
    observation_shape: Shape,
    action_size: int,
    scale: float,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
) -> DeterministicResidualPolicy:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder
    )
    policy = DeterministicResidualPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
        scale=scale,
    )
    policy.to(device)
    if enable_ddp:
        policy = wrap_model_by_ddp(policy)
    return policy


def create_normal_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
    use_std_parameter: bool = False,
) -> NormalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    policy = NormalPolicy(
        encoder=encoder,
        hidden_size=hidden_size,
        action_size=action_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
        use_std_parameter=use_std_parameter,
    )
    policy.to(device)
    if enable_ddp:
        policy = wrap_model_by_ddp(policy)
    return policy


def create_categorical_policy(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
    embedding_size: Optional[int] = None
) -> CategoricalPolicy:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder, embedding_size)
    policy = CategoricalPolicy(
        encoder=encoder, hidden_size=hidden_size, action_size=action_size
    )
    policy.to(device)
    if enable_ddp:
        policy = wrap_model_by_ddp(policy)
    return policy


def create_vae_encoder(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
    min_logstd: float = -20.0,
    max_logstd: float = 2.0,
) -> VAEEncoder:
    encoder = encoder_factory.create_with_action(observation_shape, action_size)
    encoder_hidden_size = compute_output_size(
        [observation_shape, (action_size,)], encoder
    )
    vae_encoder = VAEEncoder(
        encoder=encoder,
        hidden_size=encoder_hidden_size,
        latent_size=latent_size,
        min_logstd=min_logstd,
        max_logstd=max_logstd,
    )
    vae_encoder.to(device)
    if enable_ddp:
        vae_encoder = wrap_model_by_ddp(vae_encoder)
    return vae_encoder


def create_vae_decoder(
    observation_shape: Shape,
    action_size: int,
    latent_size: int,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
) -> VAEDecoder:
    encoder = encoder_factory.create_with_action(observation_shape, latent_size)
    decoder_hidden_size = compute_output_size(
        [observation_shape, (latent_size,)], encoder
    )
    decoder = VAEDecoder(
        encoder=encoder,
        hidden_size=decoder_hidden_size,
        action_size=action_size,
    )
    decoder.to(device)
    if enable_ddp:
        decoder = wrap_model_by_ddp(decoder)
    return decoder


def create_value_function(
    observation_shape: Shape,
    encoder_factory: EncoderFactory,
    device: str,
    enable_ddp: bool,
) -> ValueFunction:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)
    value_func = ValueFunction(encoder, hidden_size)
    value_func.to(device)
    if enable_ddp:
        value_func = wrap_model_by_ddp(value_func)
    return value_func


def create_parameter(
    shape: Sequence[int], initial_value: float, device: str, enable_ddp: bool
) -> Parameter:
    data = torch.full(shape, initial_value, dtype=torch.float32)
    parameter = Parameter(data)
    parameter.to(device)
    if enable_ddp:
        parameter = wrap_model_by_ddp(parameter)
    return parameter


def _create_position_encoding(
    position_encoding_type: PositionEncodingType,
    embed_dim: int,
    max_timestep: int,
    context_size: int,
) -> PositionEncoding:
    if position_encoding_type == PositionEncodingType.SIMPLE:
        position_encoding = SimplePositionEncoding(embed_dim, max_timestep + 1)
    elif position_encoding_type == PositionEncodingType.GLOBAL:
        position_encoding = GlobalPositionEncoding(
            embed_dim, max_timestep + 1, context_size
        )
    else:
        raise ValueError(
            f"invalid position_encoding_type: {position_encoding_type}"
        )
    return position_encoding


def create_continuous_decision_transformer(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    attn_dropout: float,
    resid_dropout: float,
    embed_dropout: float,
    activation_type: str,
    position_encoding_type: PositionEncodingType,
    device: str,
    enable_ddp: bool,
) -> ContinuousDecisionTransformer:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder)

    position_encoding = _create_position_encoding(
        position_encoding_type=position_encoding_type,
        embed_dim=hidden_size,
        max_timestep=max_timestep + 1,
        context_size=context_size,
    )

    transformer = ContinuousDecisionTransformer(
        encoder=encoder,
        embed_size=hidden_size,
        position_encoding=position_encoding,
        action_size=action_size,
        num_heads=num_heads,
        context_size=context_size,
        num_layers=num_layers,
        attn_dropout=attn_dropout,
        resid_dropout=resid_dropout,
        embed_dropout=embed_dropout,
        activation=create_activation(activation_type),
    )
    transformer.to(device)
    if enable_ddp:
        transformer = wrap_model_by_ddp(transformer)
    return transformer


def create_discrete_decision_transformer(
    observation_shape: Shape,
    action_size: int,
    encoder_factory: EncoderFactory,
    num_heads: int,
    max_timestep: int,
    num_layers: int,
    context_size: int,
    attn_dropout: float,
    resid_dropout: float,
    embed_dropout: float,
    activation_type: str,
    embed_activation_type: str,
    position_encoding_type: PositionEncodingType,
    device: str,
    enable_ddp: bool,
    embedding_size: Optional[int] = None,
) -> DiscreteDecisionTransformer:
    encoder = encoder_factory.create(observation_shape)
    hidden_size = compute_output_size([observation_shape], encoder, embedding_size)

    position_encoding = _create_position_encoding(
        position_encoding_type=position_encoding_type,
        embed_dim=hidden_size,
        max_timestep=max_timestep + 1,
        context_size=context_size,
    )

    transformer = DiscreteDecisionTransformer(
        encoder=encoder,
        embed_size=hidden_size,
        position_encoding=position_encoding,
        action_size=action_size,
        num_heads=num_heads,
        context_size=context_size,
        num_layers=num_layers,
        attn_dropout=attn_dropout,
        resid_dropout=resid_dropout,
        embed_dropout=embed_dropout,
        activation=create_activation(activation_type),
        embed_activation=create_activation(embed_activation_type),
    )
    transformer.to(device)
    if enable_ddp:
        transformer = wrap_model_by_ddp(transformer)
    return transformer
