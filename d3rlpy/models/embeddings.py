import dataclasses
from typing import Dict

from torch import nn

from ..serializable_config import DynamicConfig, generate_config_registration
from ..tokenizers import FloatTokenizer
from .torch import TokenEmbedding, TokenEmbeddingWithTokenizer

__all__ = [
    "EmbeddingModuleFactory",
    "DiscreteTokenEmbeddingModuleFactory",
    "register_embedding_module_factory",
    "make_embedding_module_field",
    "TokenEmbeddingFactory",
    "FloatTokenEmbeddingFactory",
    "register_token_embedding_factory",
    "make_token_embedding_field",
]


class EmbeddingModuleFactory(DynamicConfig):
    def create(self) -> nn.Module:
        raise NotImplementedError


@dataclasses.dataclass()
class DiscreteTokenEmbeddingModuleFactory(EmbeddingModuleFactory):
    vocab_size: int
    embed_size: int

    def create(self) -> nn.Module:
        return nn.Embedding(self.vocab_size, self.embed_size)

    @staticmethod
    def get_type() -> str:
        return "discrete_token_embedding_module"


(
    register_embedding_module_factory,
    make_embedding_module_field,
) = generate_config_registration(EmbeddingModuleFactory)
register_embedding_module_factory(DiscreteTokenEmbeddingModuleFactory)


@dataclasses.dataclass()
class TokenEmbeddingFactory(DynamicConfig):
    embedding_module_key: str

    def create(self, embedding_modules: Dict[str, nn.Module]) -> TokenEmbedding:
        return self.inner_create(embedding_modules[self.embedding_module_key])

    def inner_create(self, embedding: nn.Module) -> TokenEmbedding:
        raise NotImplementedError


@dataclasses.dataclass()
class FloatTokenEmbeddingFactory(TokenEmbeddingFactory):
    num_bins: int
    use_mu_law_encode: bool
    mu: float = 100
    basis: float = 256
    token_offset: int = 0

    def inner_create(self, embedding: nn.Module) -> TokenEmbeddingWithTokenizer:
        assert isinstance(embedding, nn.Embedding)
        tokenizer = FloatTokenizer(
            num_bins=self.num_bins,
            use_mu_law_encode=self.use_mu_law_encode,
            mu=self.mu,
            basis=self.basis,
            token_offset=self.token_offset,
        )
        return TokenEmbeddingWithTokenizer(embedding, tokenizer)

    @staticmethod
    def get_type() -> str:
        return "float_token_embedding"


(
    register_token_embedding_factory,
    make_token_embedding_field,
) = generate_config_registration(TokenEmbeddingFactory)
register_token_embedding_factory(FloatTokenEmbeddingFactory)
