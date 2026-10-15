import triton
import torch
import tempfile
import os
import subprocess
import sys
import traceback
import textwrap
from pathlib import Path
from .core import *
from typing import List, Tuple, Optional
from tqdm import tqdm


class TritonProfiler:
    def __init__(
        self,
        warmup_iters: int = 16,
        profile_iters: int = 1000,
        debug: bool = False,
        save_codes: bool = False,
    ):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.debug = debug
        self.success_num = 0
        self.fail_num = 0
        self.cnt = 0
        self.save_codes = save_codes
        self.log_dir = None

    @staticmethod
    def _split_program(code: str) -> tuple[str, str]:
        definitions, separator, main = code.partition('if __name__ == "__main__":')
        if not separator:
            raise ValueError("Generated Triton program has no main section")
        return definitions, textwrap.indent(textwrap.dedent(main).strip(), "        ")

    def _generate_profile_code_file(self, graph, target_cc: int) -> Tuple[str, str]:
        """Generate triton code for a graph and save to temp file"""

        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            result = generate_triton_program(graph.cygraph, target_cc=target_cc)
            code = result["code"]
            output_shapes = result["output_shapes"]

            definitions, main_code = self._split_program(code)

            run_kernel_code = f"""
from tqdm import tqdm

# Keep progress separate from stderr, which the parent saves to a log.
_progress_fd = os.environ.get("MIRAGE_TRITON_PROGRESS_FD")
_progress_stream = (os.fdopen(int(_progress_fd), "w", buffering=1, closefd=False)
                    if _progress_fd is not None else sys.stderr)

def profile_kernel():
    # Warmup iterations
    for _ in tqdm(range({self.warmup_iters}),
                  desc="Graph {self.cnt}: warmup / first compile",
                  file=_progress_stream, position=1, leave=False,
                  dynamic_ncols=True, mininterval=0.5):
{main_code}
    
    # Profile iterations
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    for _ in tqdm(range({self.profile_iters}),
                  desc="Graph {self.cnt}: profiling",
                  file=_progress_stream, position=1, leave=False,
                  dynamic_ncols=True, mininterval=0.5):
{main_code}
    ender.record()
    torch.cuda.synchronize()
    elapsed_time = starter.elapsed_time(ender) / {self.profile_iters}
    print(f'Average time per iteration: {{elapsed_time:.3f}} ms')

if __name__ == "__main__":
    print("Running kernel...")
    profile_kernel()

"""
            final_code = definitions + run_kernel_code

            if self.save_codes:
                Path("generated_codes").mkdir(parents=True, exist_ok=True)
                with open(f"generated_codes/generated_code_{self.cnt}.py", "w") as _f:
                    _f.write(final_code)
            f.write(final_code)
            return f.name, code, output_shapes

    def _generate_debug_code_file(self, graph, target_cc: int) -> Tuple[str, str]:
        """Generate triton code for a graph and save to temp file"""
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            header = """
import sys
import traceback
"""
            error_handling = """
def run_with_debug():
    # Enable CUDA debug mode
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    
    try:
"""
            result = generate_triton_program(graph.cygraph, target_cc=target_cc)
            code = result["code"]
            output_shapes = result["output_shapes"]

            definitions, main_code = self._split_program(code)

            debug_code = """
        print("Debug: Initializing CUDA device")
        device = torch.device('cuda')
        print(f"Debug: Using device: {device}")
        print(f"Debug: CUDA available: {torch.cuda.is_available()}")
        print(f"Debug: Current device: {torch.cuda.current_device()}")
        print(f"Debug: Device name: {torch.cuda.get_device_name(0)}")
"""

            error_handling_end = """
    except Exception as e:
        print("ERROR: " + str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_with_debug()
"""
            final_file = (
                header
                + definitions
                + error_handling
                + debug_code
                + main_code
                + error_handling_end
            )
            f.write(final_file)
            with open(f"generated_debug_code_{self.cnt}.py", "w") as _f:
                _f.write(final_file)
            return f.name, code, output_shapes

    def _log_path(self) -> Path:
        if self.log_dir is None:
            root = Path("work") / "triton-profile"
            root.mkdir(parents=True, exist_ok=True)
            self.log_dir = Path(tempfile.mkdtemp(prefix="run-", dir=root))
        return self.log_dir / f"graph-{self.cnt}.log"

    def _skip_failed_graph(self) -> None:
        log_path = self._log_path()
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        self.fail_num += 1
        tqdm.write(f"Graph {self.cnt}: skipped — details: {log_path}")

    def _run_profile(self, module_path: str) -> float:
        """Stream progress separately while saving child diagnostics to disk."""
        log_path = self._log_path()
        # This CUDA/Linux backend can inherit a separate descriptor for tqdm,
        # leaving stderr (including native compiler output) entirely in the log.
        progress_fd = os.dup(2)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                print(f"Module: {module_path}", file=log, flush=True)
                process = subprocess.run(
                    [sys.executable, module_path],
                    stdout=subprocess.PIPE,
                    stderr=log,
                    env={**os.environ, "MIRAGE_TRITON_PROGRESS_FD": str(progress_fd)},
                    pass_fds=(progress_fd,),
                    text=True,
                    check=False,
                )
                print(f"\nExit code: {process.returncode}\nStdout:\n{process.stdout}",
                      file=log)
            process.check_returncode()

            for line in process.stdout.splitlines():
                if "Average time per iteration" in line:
                    perf = float(line.split(":")[-1].strip().split(" ")[0])
                    self.success_num += 1
                    return perf
            raise RuntimeError("Failed to extract timing from output")
        except Exception:
            self._skip_failed_graph()
            return -1
        finally:
            os.close(progress_fd)

    def profile_graphs(
        self, graphs: List, target_cc: int = 10
    ) -> Tuple[object, float, List[Tuple[object, float, str]]]:
        """Profile multiple graphs and return best one with timing results

        Returns:
            Tuple containing:
            - Best performing graph
            - Best performance time
            - List of (graph, performance, code) tuples for all graphs
        """
        results = []
        best_graph = None
        best_perf = float("inf")
        best_file_path = None
        best_output_shapes = None

        print(f"Profiling {len(graphs)} candidate graphs...")

        with tqdm(total=len(graphs), desc="Processing graphs", position=0) as pbar:
            for idx, g in enumerate(graphs):
                try:
                    # Generate and save code
                    if self.debug:
                        code_path, code, output_shapes = self._generate_debug_code_file(
                            g, target_cc
                        )
                    else:
                        code_path, code, output_shapes = (
                            self._generate_profile_code_file(g, target_cc)
                        )

                    # Profile the graph with progress description
                    perf = self._run_profile(code_path)
                    if perf < 0:
                        continue

                    tqdm.write(f"Graph {idx}: {perf:.3f} ms")

                    results.append((g, perf, code))

                    if perf < best_perf and perf > 0:
                        best_graph = g
                        best_perf = perf
                        best_file_path = code_path
                        best_output_shapes = output_shapes
                        tqdm.write(f"New best performance: {perf:.3f} ms")

                except Exception:
                    self._skip_failed_graph()
                    continue

                finally:
                    self.cnt += 1
                    pbar.update(1)

        print(f"\nBest performance: {best_perf:.3f} ms")
        print(f"Successes: {self.success_num}, Failures: {self.fail_num}")
        return best_graph, best_file_path, best_output_shapes


def profile_and_select_best_graph(
    graphs: List,
    target_cc: int = 10,
    warmup_iters: int = 16,
    profile_iters: int = 1000,
    debug_mode: bool = False,
    save_codes: bool = False,
) -> object:
    """Helper function to profile graphs and select the best one"""
    profiler = TritonProfiler(warmup_iters, profile_iters, debug_mode, save_codes)
    best_graph, best_file_path, best_output_shapes = profiler.profile_graphs(
        graphs, target_cc
    )
    return best_graph, best_file_path, best_output_shapes
