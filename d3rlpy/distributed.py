import dataclasses

import torch.distributed as dist

from .logging import set_log_context

__all__ = ["init_process_group", "destroy_process_group"]


@dataclasses.dataclass(frozen=True)
class DistributedWorkerInfo:
    rank: int
    backend: str
    world_size: int


def init_process_group(backend: str) -> int:
    """Initializes process group of distributed workers.

    Internally, distributed worker information is injected to log outputs.

    Args:
        backend: Backend of communication. Available options are ``gloo``,
            ``mpi`` and ``nccl``.

    Returns:
        Rank of the current process.
    """
    dist.init_process_group(backend)
    rank = dist.get_rank()
    set_log_context(
        distributed=DistributedWorkerInfo(
            rank=dist.get_rank(),
            backend=dist.get_backend(),
            world_size=dist.get_world_size(),
        )
    )
    return int(rank)


def destroy_process_group() -> None:
    """Destroys process group of distributed workers."""
    dist.destroy_process_group()
