import math
from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from ...torch_utility import GEGLU
from ...types import TorchObservation
from .encoders import Encoder
from .parameters import Parameter

__all__ = [
    "ContinuousDecisionTransformer",
    "DiscreteDecisionTransformer",
    "PositionEncoding",
    "SimplePositionEncoding",
    "GlobalPositionEncoding",
    "GatoTransformer",
]


def create_attention_mask(context_size: int) -> torch.Tensor:
    mask = torch.ones(context_size, context_size, dtype=torch.float32)
    return torch.tril(mask).view(1, 1, context_size, context_size)


class CausalSelfAttention(nn.Module):  # type: ignore
    _num_heads: int
    _context_size: int
    _k: nn.Linear
    _q: nn.Linear
    _v: nn.Linear
    _proj: nn.Linear
    _attn_dropout: nn.Dropout
    _proj_dropout: nn.Dropout
    _mask: torch.Tensor

    def __init__(
        self,
        embed_size: int,
        num_heads: int,
        context_size: int,
        attn_dropout: float,
        resid_dropout: float,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._context_size = context_size
        self._k = nn.Linear(embed_size, embed_size)
        self._q = nn.Linear(embed_size, embed_size)
        self._v = nn.Linear(embed_size, embed_size)
        self._proj = nn.Linear(embed_size, embed_size)
        self._attn_dropout = nn.Dropout(attn_dropout)
        self._proj_dropout = nn.Dropout(resid_dropout)
        mask = create_attention_mask(context_size)
        self.register_buffer("_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expects (B, T, N), but got {x.shape}"
        batch_size, context_size, _ = x.shape
        assert context_size <= self._context_size, "Exceeds context_size"

        # (B, T, N) -> (B, T, H, N / H) -> (B, H, T, N / H)
        shape = (batch_size, context_size, self._num_heads, -1)
        k = self._k(x).view(shape).transpose(1, 2)
        q = self._q(x).view(shape).transpose(1, 2)
        v = self._v(x).view(shape).transpose(1, 2)

        # (B, H, T, N / H) -> (B, H, T, T)
        qkT = torch.matmul(q, k.transpose(2, 3))
        attention = qkT / math.sqrt(k.shape[-1])
        attention = attention.masked_fill(
            self._mask[..., :context_size, :context_size] == 0, float("-inf")
        )
        attention = F.softmax(attention, dim=-1)
        attention = self._attn_dropout(attention)

        # (B, H, T, T) x (B, H, T, N / H) -> (B, H, T, N / H)
        output = torch.matmul(attention, v)
        # (B, H, T, N / H) -> (B, T, N)
        output = output.transpose(1, 2).reshape(batch_size, context_size, -1)

        return self._proj_dropout(self._proj(output))


class MLP(nn.Module):  # type: ignore
    _l1: nn.Linear
    _l2: nn.Linear
    _dropout: nn.Dropout
    _activation: nn.Module

    def __init__(
        self,
        in_size: int,
        out_size: int,
        pre_activation_hidden_size: int,
        post_activation_hidden_size: int,
        dropout: float,
        activation: nn.Module,
    ):
        super().__init__()
        self._l1 = nn.Linear(in_size, pre_activation_hidden_size)
        self._l2 = nn.Linear(post_activation_hidden_size, out_size)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._activation(self._l1(x))
        h = self._dropout(self._l2(h))
        return h


class Block(nn.Module):  # type: ignore
    _attention: CausalSelfAttention
    _mlp: MLP
    _layer_norm1: nn.LayerNorm
    _layer_norm2: nn.LayerNorm

    def __init__(
        self,
        layer_width: int,
        pre_activation_ff_hidden_size: int,
        post_activation_ff_hidden_size: int,
        num_heads: int,
        context_size: int,
        attn_dropout: float,
        resid_dropout: float,
        activation: nn.Module,
    ):
        super().__init__()
        self._attention = CausalSelfAttention(
            embed_size=layer_width,
            num_heads=num_heads,
            context_size=context_size,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )
        self._mlp = MLP(
            in_size=layer_width,
            out_size=layer_width,
            pre_activation_hidden_size=pre_activation_ff_hidden_size,
            post_activation_hidden_size=post_activation_ff_hidden_size,
            dropout=resid_dropout,
            activation=activation,
        )
        self._layer_norm1 = nn.LayerNorm(layer_width, eps=0.003)
        self._layer_norm2 = nn.LayerNorm(layer_width, eps=0.003)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self._layer_norm1(x)
        x = x + self._attention(norm_x)
        norm_x = self._layer_norm2(x)
        x = x + self._mlp(norm_x)
        return x


class PositionEncoding(nn.Module, metaclass=ABCMeta):  # type: ignore
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SimplePositionEncoding(PositionEncoding):
    def __init__(self, embed_dim: int, max_timestep: int):
        super().__init__()
        self._embed = nn.Embedding(max_timestep, embed_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 2, "Expects (B, T)"
        # (B, T) -> (B, T, N)
        return self._embed(t)


class GlobalPositionEncoding(PositionEncoding):
    def __init__(self, embed_dim: int, max_timestep: int, context_size: int):
        super().__init__()
        self._embed_dim = embed_dim
        self._global_position_embedding = Parameter(
            torch.zeros(1, max_timestep, embed_dim, dtype=torch.float32)
        )
        self._block_position_embedding = Parameter(
            torch.zeros(1, 3 * context_size, embed_dim, dtype=torch.float32)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 2, "Expects (B, T)"
        batch_size, context_size = t.shape

        # (B, 1, 1) -> (B, 1, N)
        last_t = torch.repeat_interleave(
            t[:, -1].view(-1, 1, 1), self._embed_dim, dim=-1
        )
        # (1, Tmax, N) -> (B, Tmax, N)
        batched_global_embedding = torch.repeat_interleave(
            self._global_position_embedding(),
            batch_size,
            dim=0,
        )
        # (B, Tmax, N) -> (B, 1, N)
        global_embedding = torch.gather(batched_global_embedding, 1, last_t)

        # (1, 3 * Cmax, N) -> (1, T, N)
        block_embedding = self._block_position_embedding()[:, :context_size, :]

        # (B, 1, N) + (1, T, N) -> (B, T, N)
        return global_embedding + block_embedding


class GPT2(nn.Module):  # type: ignore
    _transformer: nn.Sequential
    _layer_norm: nn.LayerNorm
    _dropout: nn.Dropout

    def __init__(
        self,
        layer_width: int,
        pre_activation_ff_hidden_size: int,
        post_activation_ff_hidden_size: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        embed_dropout: float,
        activation: nn.Module,
    ):
        super().__init__()
        blocks = [
            Block(
                layer_width=layer_width,
                pre_activation_ff_hidden_size=pre_activation_ff_hidden_size,
                post_activation_ff_hidden_size=post_activation_ff_hidden_size,
                num_heads=num_heads,
                context_size=context_size,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ]
        self._transformer = nn.Sequential(*blocks)
        self._layer_norm = nn.LayerNorm(layer_width, eps=0.003)
        self._dropout = nn.Dropout(embed_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._dropout(x)
        h = self._transformer(h)
        h = self._layer_norm(h)
        return h


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class ContinuousDecisionTransformer(nn.Module):  # type: ignore
    _encoder: Encoder
    _position_encoding: PositionEncoding
    _action_embed: nn.Linear
    _rtg_embed: nn.Linear
    _gpt2: GPT2
    _output: nn.Linear

    def __init__(
        self,
        encoder: Encoder,
        embed_size: int,
        position_encoding: PositionEncoding,
        action_size: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        embed_dropout: float,
        activation: nn.Module,
    ):
        super().__init__()
        self._position_encoding = position_encoding
        self._embed_ln = nn.LayerNorm(embed_size)
        self._gpt2 = GPT2(
            layer_width=embed_size,
            pre_activation_ff_hidden_size=4 * embed_size,
            post_activation_ff_hidden_size=4 * embed_size,
            num_heads=num_heads,
            context_size=3 * context_size,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            activation=activation,
        )
        self.apply(_init_weights)

        self._encoder = encoder
        self._rtg_embed = nn.Linear(1, embed_size)
        self._action_embed = nn.Linear(action_size, embed_size)
        self._output = nn.Linear(embed_size, action_size)

    def forward(
        self,
        x: TorchObservation,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, context_size, _ = return_to_go.shape
        position_embedding = self._position_encoding(timesteps)

        if isinstance(x, torch.Tensor):
            flat_x = x.view(-1, *x.shape[2:])
        else:
            flat_x = [_x.view(-1, *_x.shape[2:]) for _x in x]
        flat_state_embedding = self._encoder(flat_x)
        state_embedding = flat_state_embedding.view(
            batch_size, context_size, -1
        )
        state_embedding = state_embedding + position_embedding

        action_embedding = self._action_embed(action) + position_embedding
        rtg_embedding = self._rtg_embed(return_to_go) + position_embedding

        # (B, T, N) -> (B, 3, T, N)
        h = torch.stack(
            [rtg_embedding, state_embedding, action_embedding], dim=1
        )
        # (B, 3, T, N) -> (B, T, 3, N) -> (B, T * 3, N)
        h = h.transpose(1, 2).reshape(batch_size, 3 * context_size, -1)

        # for inference, drop the last step action to prevent copy
        if not self.training:
            h = h[:, :-1, :]

        h = self._gpt2(self._embed_ln(h))

        return torch.tanh(self._output(h[:, 1::3, :]))


class DiscreteDecisionTransformer(nn.Module):  # type: ignore
    _encoder: Encoder
    _position_encoding: PositionEncoding
    _action_embed: nn.Embedding
    _rtg_embed: nn.Linear
    _gpt2: GPT2
    _output: nn.Linear
    _embed_activation: nn.Module

    def __init__(
        self,
        encoder: Encoder,
        embed_size: int,
        position_encoding: PositionEncoding,
        action_size: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        embed_dropout: float,
        activation: nn.Module,
        embed_activation: nn.Module,
    ):
        super().__init__()
        self._position_encoding = position_encoding
        self._gpt2 = GPT2(
            layer_width=embed_size,
            pre_activation_ff_hidden_size=4 * embed_size,
            post_activation_ff_hidden_size=4 * embed_size,
            num_heads=num_heads,
            context_size=3 * context_size,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            activation=activation,
        )
        self._output = nn.Linear(embed_size, action_size, bias=False)
        self._action_embed = nn.Embedding(action_size, embed_size)
        self.apply(_init_weights)

        self._encoder = encoder
        self._rtg_embed = nn.Linear(1, embed_size)
        self._embed_activation = embed_activation

    def forward(
        self,
        x: TorchObservation,
        action: torch.Tensor,
        return_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_size, _ = return_to_go.shape
        position_embedding = self._position_encoding(timesteps)

        if isinstance(x, torch.Tensor):
            flat_x = x.reshape(-1, *x.shape[2:])
        else:
            flat_x = [_x.reshape(-1, *_x.shape[2:]) for _x in x]
        flat_state_embedding = self._encoder(flat_x)
        state_embedding = flat_state_embedding.view(
            batch_size, context_size, -1
        )
        flat_action = action.view(batch_size, context_size).long()
        action_embedding = self._action_embed(flat_action)
        rtg_embedding = self._rtg_embed(return_to_go)

        # (B, T, N) -> (B, 3, T, N)
        h = torch.stack(
            [rtg_embedding, state_embedding, action_embedding], dim=1
        )
        h = self._embed_activation(h)
        h = h + position_embedding.view(batch_size, 1, context_size, -1)
        # (B, 3, T, N) -> (B, T, 3, N) -> (B, T * 3, N)
        h = h.transpose(1, 2).reshape(batch_size, 3 * context_size, -1)

        # for inference, drop the last step action to prevent copy
        if not self.training:
            h = h[:, :-1, :]

        h = self._gpt2(h)

        # use state embeddings as input
        logits = self._output(h[:, 1::3, :])

        return F.softmax(logits, dim=-1), logits


class GatoTransformer(nn.Module):  # type: ignore
    _gpt2: GPT2
    _token_embed: nn.Embedding
    _observation_pos_embed: nn.Embedding
    _action_pos_embed: Parameter
    _output: nn.Linear
    _embed_activation: nn.Module

    def __init__(
        self,
        layer_width: int,
        ff_hidden_size: int,
        max_observation_length: int,
        vocab_size: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
        attn_dropout: float,
        resid_dropout: float,
        embed_dropout: float,
        embed_activation: nn.Module,
    ):
        super().__init__()
        self._gpt2 = GPT2(
            layer_width=layer_width,
            pre_activation_ff_hidden_size=2 * ff_hidden_size,
            post_activation_ff_hidden_size=ff_hidden_size,
            num_heads=num_heads,
            context_size=context_size,
            num_layers=num_layers,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            embed_dropout=embed_dropout,
            activation=GEGLU(),
        )
        self._output = nn.Linear(layer_width, vocab_size, bias=False)
        # +1 for separator token
        self._token_embed = nn.Embedding(vocab_size + 1, layer_width)
        self._observation_pos_embed = nn.Embedding(
            max_observation_length, layer_width
        )
        self._action_pos_embed = Parameter(
            torch.zeros(1, 1, layer_width, dtype=torch.float32)
        )
        self.apply(_init_weights)
        self._embed_activation = embed_activation

    def forward(
        self,
        tokens: torch.Tensor,
        observation_masks: torch.Tensor,
        observation_positions: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: Support text and patch tokens
        assert tokens.ndim == 2
        batch_size, context_size = tokens.shape
        assert observation_masks.shape == (batch_size, context_size, 1)
        assert observation_positions.shape == (batch_size, context_size)
        assert action_masks.shape == (batch_size, context_size, 1)

        # (B, T, N)
        embeddings = self._embed_activation(self._token_embed(tokens))

        # add local observation embedding
        embeddings = (
            embeddings
            + observation_masks
            * self._observation_pos_embed(observation_positions)
        )

        # add action embedding
        embeddings = embeddings + action_masks * self._action_pos_embed()

        # (B, T, N) -> (B, T, N)
        h = self._gpt2(embeddings)

        # (B, T, N) -> (B, T, vocab)
        logits = self._output(h)

        return F.softmax(logits, dim=-1), logits
