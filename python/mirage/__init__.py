"""Mirage Persistent Kernel: graph construction, compilation, and runtime."""
from . import core
from .core import *
from .kernel import *
from .mpk.persistent_kernel import PersistentKernel
from .mpk.speculative import spec_decode_class
# Re-export MPK interfaces for external integrations (e.g., vLLM Mirage backend)
from .mpk.mpk import MPK, MPKMetadata
from .mpk.models.graph_builder import MirageModelConfig
from .threadblock import *


class InputNotFoundError(Exception):
    """Raised when cannot find input tensors"""

    pass


def set_gpu_device_id(device_id: int):
    global_config.gpu_device_id = device_id
    core.set_gpu_device_id(device_id)


def new_kernel_graph():
    kgraph = core.CyKNGraph()
    return KNGraph(kgraph)


def new_threadblock_graph(
    grid_dim: tuple, block_dim: tuple, forloop_range: int, reduction_dimx: int
):
    bgraph = core.CyTBGraph(grid_dim, block_dim, forloop_range, reduction_dimx)
    return TBGraph(bgraph)


# Other Configurations
from .global_config import global_config

from .version import __version__