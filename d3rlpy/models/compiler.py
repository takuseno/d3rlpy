import dataclasses
from typing import Any, Dict, Optional, Sequence, TypeVar

import torch

from ..dataset import Shape
from ..serializable_config import (
    DynamicConfig,
    generate_optional_config_generation,
)

__all__ = ["Compiler", "TraceCompiler", "TorchCompiler", "make_compiler_field", "create_example_input"]


T = TypeVar("T")


def create_example_input_shape(shape: Sequence[int], batch_size: int = 1, seq_size: Optional[int] = None) -> Sequence[int]:
    if seq_size is None:
        return (batch_size, *shape)
    else:
        return (batch_size, seq_size, *shape)


def create_example_input(shape: Shape, batch_size: int = 1, seq_size: Optional[int] = None) -> torch.Tensor:
    if isinstance(shape[0], (list, tuple)):
        return [torch.rand(create_example_input_shape(s, batch_size, seq_size), dtype=torch.float32) for s in shape]
    else:
        return torch.rand(create_example_input_shape(shape, batch_size, seq_size), dtype=torch.float32)


class Compiler(DynamicConfig):
    """A base class of compilers."""

    def compile(self, model: T, example_inputs: Sequence[torch.Tensor]) -> T:
        """Returns a compiled model.

        Args:
            model: nn.Module object to compile.
            example_inputs: Example inputs to trace.

        Returns:
            A compiled model.
        """
        raise NotImplementedError


@dataclasses.dataclass()
class TraceCompiler(Compiler):
    """A wrapper class of torch.jit.trace.

    See references for details.

    References:
        * `PyTorch documentation.
          <https://pytorch.org/docs/stable/generated/torch.jit.trace.html>`_
        * `TorchScript Tutorial.
          <https://pytorch.org/docs/stable/jit.html>`_
    """
    def compile(self, model: T, example_inputs: Sequence[torch.Tensor]) -> T:
        return torch.jit.trace(model, example_inputs=example_inputs)

    @staticmethod
    def get_type() -> str:
        return "torch.jit.trace"


@dataclasses.dataclass()
class TorchCompiler(Compiler):
    """A wrapper class of torch.compile.

    See references for details.

    References:
        * `PyTorch documentation.
          <https://pytorch.org/docs/stable/generated/torch.compile.html>`_
        * `torch.compile Tutorial.
          <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>`_

    Args:
        fullgraph (bool): Whether it is ok to break model into several
            subgraphs.
        mode (str): Can be either ``default``, ``reduce-overhead`` or
            ``max-autotune``.
        backend (str): Backend to be used.
        options (Optional[Dict[str, Any]]): A dictionary of options to pass to
            the backend.
    """

    fullgraph: bool = False
    mode: str = "default"
    backend: str = "inductor"
    options: Optional[Dict[str, Any]] = None

    def compile(self, model: T, example_inputs: Sequence[torch.Tensor]) -> T:
        return torch.compile(  # type: ignore
            model,
            fullgraph=self.fullgraph,
            mode=self.mode,
            options=self.options,
        )

    @staticmethod
    def get_type() -> str:
        return "torch.compile"


register_compiler, make_compiler_field = generate_optional_config_generation(
    Compiler,
)

register_compiler(TorchCompiler)
