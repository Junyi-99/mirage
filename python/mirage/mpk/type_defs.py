"""Static contracts for MPK configuration and opaque Cython tensor handles."""

from typing import TYPE_CHECKING, Protocol, TypedDict

import torch

if TYPE_CHECKING:
    from .kv_planner import KVGroupSpec
    from .speculative import SpecDecodeConfig


class TensorDescriptor(Protocol):
    """Shape interface of the Cython DTensor; a null handle returns None."""

    @property
    def num_dims(self) -> int | None: ...

    def dim(self, idx: int) -> int | None: ...


class _RequiredInitParameters(TypedDict):
    mode: str
    world_size: int
    mpi_rank: int
    num_workers: int
    num_local_schedulers: int
    num_remote_schedulers: int
    max_seq_length: int
    max_num_batched_requests: int
    max_num_batched_tokens: int
    max_num_pages: int
    meta_tensors: dict[str, torch.Tensor]
    profiler_tensor: torch.Tensor | None
    trace_name: str
    spec_decode_config: "SpecDecodeConfig | None"
    use_cutlass_kernel: bool


class InitParameters(_RequiredInitParameters, total=False):
    eos_token_id: int
    pinned_ring_capacity: int
    test_mode: bool
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int
    sampling_seed: int
    sampling_topk_max: int
    kv_groups: list["KVGroupSpec"] | None
    page_size: int | None
    max_tokens_per_request: int | None

