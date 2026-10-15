"""MPK-only API and GPU compilation regression tests."""
import subprocess
import sys

import pytest
import torch
import mirage as mi


def test_import_without_search_dependencies():
    subprocess.run([sys.executable, '-c', '''
import importlib.abc, sys
class NoZ3(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'z3':
            raise ImportError('Z3 is forbidden in MPK-only')
sys.meta_path.insert(0, NoZ3())
import mirage as mi
assert callable(mi.PersistentKernel.compile)
assert not hasattr(mi.KNGraph, 'superoptimize')
for name in ('search', 'generate_cuda_program', 'generate_triton_program', 'generate_nki_program'):
    assert not hasattr(mi, name), name
'''], check=True)


def test_graph_serialization(tmp_path):
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(1, 4096), dtype=mi.bfloat16)
    graph.mark_output(graph.silu(x))
    filename = str(tmp_path / 'graph.json')
    graph.to_json(filename)
    restored = mi.new_kernel_graph()
    restored.from_json(filename)
    assert restored.get_owner_independent_hash() == graph.get_owner_independent_hash()
    with pytest.raises(RuntimeError, match='Cannot open'):
        restored.from_json(str(tmp_path / 'missing.json'))


def test_unsupported_graph_serialization(tmp_path):
    graph = mi.new_kernel_graph()
    x = graph.new_input(dims=(1, 4096), dtype=mi.bfloat16)
    graph.mark_output(graph.rms_norm(x, normalized_shape=(4096,)))
    filename = str(tmp_path / 'unsupported.json')
    graph.to_json(filename)
    with pytest.raises(RuntimeError, match='Cannot deserialize'):
        mi.new_kernel_graph().from_json(filename)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU required')
def test_dependent_rmsnorm(tmp_path):
    from mirage.utils import get_configurations_from_gpu
    generator = torch.Generator(device='cuda').manual_seed(17)
    x = (torch.randn((1, 4096), device='cuda', generator=generator) * 7).bfloat16()
    w1 = torch.randn(4096, device='cuda', dtype=x.dtype, generator=generator)
    w2 = torch.randn(4096, device='cuda', dtype=x.dtype, generator=generator)
    intermediate, output = torch.full_like(x, float('nan')), torch.full_like(x, float('nan'))
    workers, schedulers = get_configurations_from_gpu(torch.cuda.current_device())
    params = mi.PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=workers, num_local_schedulers=schedulers)
    pk = mi.PersistentKernel(**params)
    try:
        dx, dw1, dw2, dt, dy = [pk.attach_input(t, name=name) for t, name in
            [(x, 'x'), (w1, 'w1'), (w2, 'w2'), (intermediate, 'tmp'), (output, 'y')]]
        launch = dict(grid_dim=(1, 1, 1), block_dim=(256 if pk.target_cc >= 90 else 128, 1, 1))
        pk.rmsnorm_layer(input=dx, weight=dw1, output=dt, **launch)
        pk.rmsnorm_layer(input=dt, weight=dw2, output=dy, **launch)
        pk.compile(output_dir=str(tmp_path / 'mpk_build'))
        pk()
        torch.cuda.synchronize()
        def reference(a, w):
            a = a.float()
            return (a * torch.rsqrt(a.square().mean(-1, keepdim=True) + 1e-6) * w.float()).to(x.dtype)
        expected_intermediate = reference(x, w1)
        expected = reference(expected_intermediate, w2)
        torch.testing.assert_close(intermediate, expected_intermediate, rtol=0.03, atol=0.03)
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.03)
    finally:
        pk.finalize()
