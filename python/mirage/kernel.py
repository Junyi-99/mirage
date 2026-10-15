"""Graph construction and task registration for Mirage Persistent Kernel."""
import os
import torch
from .core import *
from .threadblock import TBGraph
from .visualizer import visualizer


def get_key_paths():
    root_dir = os.path.join(
        os.path.dirname(__file__), "../.."
    )  # Using pip install -e .
    if not os.path.exists(os.path.join(root_dir, "deps")):  # Using pip install .
        root_dir = os.path.dirname(__file__)

    # If MIRAGE_ROOT is not set, use the root_dir as MIRAGE_ROOT
    MIRAGE_ROOT = os.environ.get("MIRAGE_ROOT", root_dir)

    INCLUDE_PATH = ""
    DEPS_PATH = ""
    if os.path.exists(os.path.join(MIRAGE_ROOT, "deps")):
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "deps")
    else:
        INCLUDE_PATH = os.path.join(MIRAGE_ROOT, "include")
        DEPS_PATH = os.path.join(MIRAGE_ROOT, "include/deps")

    assert os.path.exists(
        MIRAGE_ROOT
    ), "No MIRAGE_ROOT directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        INCLUDE_PATH
    ), "No /include directory found. Likely using the wrong MIRAGE_ROOT."
    assert os.path.exists(
        DEPS_PATH
    ), "No /deps directory found. Likely using the wrong MIRAGE_ROOT."

    return MIRAGE_ROOT, INCLUDE_PATH, DEPS_PATH


class KNGraph:
    """MPK graph builder; execution is compiled by PersistentKernel.compile()."""

    def __init__(self, graph: CyKNGraph):
        self.cygraph = graph
        self.visualizer = None

    def new_input(
        self,
        dims: tuple[int, ...],
        strides: tuple[int, ...] | None = None,
        dtype: dtype = float16,
    ) -> DTensor:
        if strides is None:
            total_elements = 1
            reversed_strides: list[int] = []
            for d in reversed(dims):
                reversed_strides.append(total_elements)
                total_elements *= d
            strides = tuple(reversed(reversed_strides))
        else:
            assert len(dims) == len(strides)
        return self.cygraph.new_input(dims, strides, dtype)

    def mark_output(self, A: DTensor, strides: tuple[int, ...] | None = None) -> None:
        return self.cygraph.mark_output(A, strides)

    def matmul(self, A: DTensor, B: DTensor) -> DTensor:
        return self.cygraph.matmul(A, B)

    def reduction(self, A: DTensor, dim: int):
        return self.cygraph.reduction(A, dim)

    def exp(self, A: DTensor):
        return self.cygraph.exp(A)

    def silu(self, A: DTensor):
        return self.cygraph.silu(A)

    def gelu(self, A: DTensor):
        return self.cygraph.gelu(A)

    def relu(self, A: DTensor):
        return self.cygraph.relu(A)

    def clamp(self, A: DTensor, min_val: float, max_val: float):
        return self.cygraph.clamp(A, min_val, max_val)

    def sqrt(self, A: DTensor):
        return self.cygraph.sqrt(A)

    def square(self, A: DTensor):
        return self.cygraph.square(A)

    def add(self, A: DTensor, B: DTensor):
        return self.cygraph.add(A, B)

    def mul(self, A: DTensor, B: DTensor):
        return self.cygraph.mul(A, B)

    def div(self, A: DTensor, B: DTensor):
        return self.cygraph.div(A, B)

    def pow(self, A: DTensor, B: DTensor):
        return self.cygraph.pow(A, B)

    def rms_norm(self, A: DTensor, normalized_shape: tuple[int, ...]) -> DTensor:
        return self.cygraph.rms_norm(A, normalized_shape)

    def customized(self, inputs: list[DTensor], bgraph: TBGraph) -> list[DTensor]:
        return self.cygraph.customized(inputs, bgraph.cygraph)

    def get_owner_independent_hash(self):
        return self.cygraph.get_owner_independent_hash()

    def visualize(self, file_name):
        operators = self.cygraph.get_graph_structure()
        self.visualizer = visualizer(file_name)
        self.visualizer.draw_graphs(operators)

    def to_json(self, filename):
        cy_to_json(self.cygraph, filename)

    def from_json(self, filename):
        self.cygraph = cy_from_json(filename)

    # Persistent Kernel functions
    def attach_torch_tensor(self, t: DTensor, torch_tensor: torch.Tensor, name: str):
        return self.cygraph.attach_torch_tensor(t, torch_tensor, name)

    def attach_cuda_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_cuda_tensor(t, name)

    def attach_nvshmem_tensor(self, t: DTensor, name: str):
        return self.cygraph.attach_nvshmem_tensor(t, name)

    def fuse_tensors(
        self, input: list[DTensor], fuse_dim: int, num_groups: int, name: str
    ):
        return self.cygraph.fuse_tensors(input, fuse_dim, num_groups, name)

    def shuffle_tensors(
        self, input: list[DTensor], shuffled_dim: int, num_groups: int, name: str
    ):
        return self.cygraph.shuffle_tensors(input, shuffled_dim, num_groups, name)

    # Virtual-tensor (view) operations
    def view(self, input: DTensor, new_shape: list) -> DTensor:
        return self.cygraph.view(input, list(new_shape))

    def narrow(self, input: DTensor, dim: int, start: int, length: int) -> DTensor:
        return self.cygraph.narrow(input, dim, start, length)

    def split(self, input: DTensor, sizes_or_chunks, dim: int) -> list:
        return self.cygraph.split(input, sizes_or_chunks, dim)

    def register_task(self, bgraph: TBGraph, task_type: str, params: list[int] = None):
        return self.cygraph.register_task(bgraph.cygraph, task_type, params)

    def generate_task_graph(self, num_gpus: int, my_gpu_id: int):
        return self.cygraph.generate_task_graph(num_gpus, my_gpu_id)
