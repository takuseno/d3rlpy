import numpy as np
import torch
from torch import nn

from ...tokenizers import Tokenizer
from ...types import Int32NDArray, NDArray
from .parameters import Parameter

__all__ = [
    "TokenEmbedding",
    "TokenEmbeddingWithTokenizer",
    "SeparatorTokenEmbedding",
]


class TokenEmbedding(nn.Module):  # type: ignore
    def __call__(self, x: NDArray) -> torch.Tensor:
        assert isinstance(x, np.ndarray)
        return super().__call__(x)

    def get_tokens(self, x: NDArray) -> Int32NDArray:
        raise NotImplementedError

    def decode(self, x: Int32NDArray) -> NDArray:
        raise NotImplementedError


class TokenEmbeddingWithTokenizer(TokenEmbedding):
    _embed: nn.Embedding
    _tokenizer: Tokenizer

    def __init__(self, embed: nn.Embedding, tokenizer: Tokenizer):
        super().__init__()
        self._embed = embed
        self._tokenizer = tokenizer

    def forward(self, x: NDArray) -> torch.Tensor:
        tokenized_x = self._tokenizer(x)
        device = next(self._embed.parameters()).device
        torch_tokenized_x = torch.tensor(
            tokenized_x, dtype=torch.int32, device=device
        )
        embedding = self._embed(torch_tokenized_x)
        assert (
            embedding.ndim == 3
        ), f"The resulted shape must be (seq, num_tokens, embed): {embedding.shape}"
        return embedding

    def get_tokens(self, x: NDArray) -> Int32NDArray:
        return self._tokenizer(x)

    def decode(self, x: Int32NDArray) -> NDArray:
        return self._tokenizer.decode(x)


class SeparatorTokenEmbedding(nn.Module):  # type: ignore
    _data: Parameter

    def __init__(self, embed_size: int):
        super().__init__()
        self._data = Parameter(torch.zeros(embed_size, dtype=torch.float32))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3
        assert x.shape[-1] == self._data.data.shape[0]
        data = self._data.data.view(1, 1, -1)
        return torch.tile(data, [x.shape[0], 1, 1])

    @property
    def data(self) -> torch.Tensor:
        return self._data.data
