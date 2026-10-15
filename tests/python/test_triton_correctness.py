"""GPU regressions for Triton code generation (run with unittest or pytest).

python -m unittest discover -s tests/python -p test_triton_correctness.py -v
"""
import importlib.util
from itertools import product
import os
from pathlib import Path
import tempfile
import unittest

import torch
import mirage as mi
from mirage.triton_profiler import TritonProfiler


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TritonCorrectnessTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.module_index = 0
        self.target_cc = sum(a * b for a, b in zip(torch.cuda.get_device_capability(), (10, 1)))

    def input(self, graph, tensor):
        dtype = mi.float32 if tensor.dtype == torch.float32 else mi.float16
        return graph.new_input(tuple(tensor.shape), tuple(tensor.stride()), dtype=dtype)

    def execute(self, graph, inputs, expected):
        result = mi.generate_triton_program(graph.cygraph, target_cc=self.target_cc)
        self.module_index += 1
        path = Path(self.directory.name) / f'generated_{self.module_index}.py'
        path.write_text(result['code'])
        spec = importlib.util.spec_from_file_location('triton_regression', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Non-contiguous destinations catch accidental rebinding and stride loss.
        outputs = [torch.full((*value.shape, 2), float('nan'), device='cuda',
                              dtype=value.dtype)[..., 0] for value in expected]
        module.execute_mugraph(*inputs, *outputs)
        torch.cuda.synchronize()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual, reference, rtol=2e-3, atol=2e-3)
        return result

    def test_unfused_binary(self):
        for shape in [(2, 256), (3, 257), (2, 3, 129), (257,)]:
            for op in ['add', 'mul', 'div', 'pow']:
                with self.subTest(shape=shape, op=op):
                    x = torch.rand((*shape, 2), device='cuda', dtype=torch.float16)[..., 0] + 0.5
                    # Use a genuine strided view after adding the positive offset.
                    x = torch.stack([x, x], dim=-1)[..., 0]
                    y = torch.rand((*shape[:-1], 1), device='cuda', dtype=x.dtype) + 0.5
                    graph = mi.new_kernel_graph()
                    a, b = self.input(graph, x), self.input(graph, y)
                    graph.mark_output(getattr(graph, op)(a, b))
                    self.execute(graph, [x, y], [getattr(torch, op)(x, y)])

    def test_unfused_unary(self):
        for dtype in [torch.float16, torch.float32]:
            for op in ['exp', 'sqrt', 'square']:
                with self.subTest(dtype=dtype, op=op):
                    x = (torch.rand((2, 3, 258), device='cuda', dtype=dtype) + 0.5)[..., ::2]
                    graph = mi.new_kernel_graph()
                    graph.mark_output(getattr(graph, op)(self.input(graph, x)))
                    self.execute(graph, [x], [getattr(torch, op)(x)])

    def test_reductions(self):
        for shape in [(2, 256), (3, 257), (2, 3, 129)]:
            for dim in range(len(shape)):
                with self.subTest(shape=shape, dim=dim):
                    x = torch.randn((*shape, 2), device='cuda', dtype=torch.float16)[..., 0]
                    graph = mi.new_kernel_graph()
                    graph.mark_output(graph.reduction(self.input(graph, x), dim))
                    self.execute(graph, [x], [x.sum(dim=dim, keepdim=True)])

    def test_intermediates_and_multiple_outputs(self):
        for dtype in [torch.float16, torch.float32]:
            with self.subTest(dtype=dtype):
                x = torch.rand((3, 257), device='cuda', dtype=dtype) + 0.5
                y = torch.rand((3, 1), device='cuda', dtype=dtype) + 0.5
                graph = mi.new_kernel_graph()
                a, b = self.input(graph, x), self.input(graph, y)
                added = graph.add(a, b)
                squared = graph.square(added)
                reduced = graph.reduction(squared, 1)
                graph.mark_output(squared)
                graph.mark_output(reduced)
                reference = (x + y).square()
                self.execute(graph, [x, y], [reference, reference.sum(1, keepdim=True)])

    def test_threadblock_add_mul(self):
        for op, iterations in product(['add', 'mul'], [1, 2]):
            with self.subTest(op=op, iterations=iterations):
                x = torch.rand((4, 128), device='cuda', dtype=torch.float16) + 0.5
                y = torch.rand_like(x) + 1.5
                graph = mi.new_kernel_graph()
                a, b = self.input(graph, x), self.input(graph, y)
                tb = mi.new_threadblock_graph((1, 1, 1), (128, 1, 1), iterations, 64)
                ta = tb.new_input(a, (-1, -1, -1), 0)
                tb_input = tb.new_input(b, (-1, -1, -1), 0)
                value = getattr(tb, op)(ta, tb_input)
                value = tb.forloop_accum(value)
                tb.new_output(value, (-1, -1, -1))
                graph.mark_output(graph.customized([a, b], tb)[0])
                # Custom kernels honor the graph's declared contiguous layout.
                result = mi.generate_triton_program(graph.cygraph, target_cc=self.target_cc)
                path = Path(self.directory.name) / f'fused_{op}_{iterations}.py'
                path.write_text(result['code'])
                spec = importlib.util.spec_from_file_location('fused_' + op, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                reference = getattr(torch, op)(x, y)
                reference = reference.reshape(iterations, 4 // iterations, 128).sum(0)
                output = torch.full_like(reference, float('nan'))
                module.execute_mugraph(x, y, output)
                torch.testing.assert_close(output, reference, rtol=2e-3, atol=2e-3)
                if op == 'add' and iterations == 2:
                    profiler = TritonProfiler(warmup_iters=1, profile_iters=2)
                    profiler.log_dir = Path(self.directory.name)
                    profile_path, _, _ = profiler._generate_profile_code_file(graph, self.target_cc)
                    self.addCleanup(Path(profile_path).unlink)
                    self.assertGreater(profiler._run_profile(profile_path), 0)

    def test_profiler_save_codes_without_custom_kernel(self):
        graph = mi.new_kernel_graph()
        x = torch.ones((2, 256), device='cuda', dtype=torch.float16)
        graph.mark_output(graph.square(self.input(graph, x)))
        previous = os.getcwd()
        try:
            os.chdir(self.directory.name)
            profiler = TritonProfiler(warmup_iters=1, profile_iters=2, save_codes=True)
            path, _, _ = profiler._generate_profile_code_file(graph, self.target_cc)
            self.addCleanup(Path(path).unlink)
            self.assertTrue(Path('generated_codes/generated_code_0.py').is_file())
            compile(Path(path).read_text(), path, 'exec')
            self.assertGreater(profiler._run_profile(path), 0)
            debug_path, _, _ = profiler._generate_debug_code_file(graph, self.target_cc)
            self.addCleanup(Path(debug_path).unlink)
            compile(Path(debug_path).read_text(), debug_path, 'exec')
        finally:
            os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
