# MPK-only branch

This branch retains graph construction, task registration, dependency/event generation,
CUDA code generation for the persistent runtime, GPU scheduling, and task profiling.
`PersistentKernel.compile()` does not run tensor-program search.

Removed:

- `KNGraph.superoptimize()`, standalone `KNGraph.compile()` and graph execution.
- Search, SMT verification, Rust symbolic-expression libraries, and Z3 dependency.
- Standalone CUDA, Triton, and NKI transpilers and their Python exports.
- Search-specific examples, benchmarks, cached graphs, and tests, including the old Qwen2.5 search demo.

Kept:

- KNGraph/TBGraph and operator implementations used to construct MPK task graphs.
- Graph JSON serialization for supported graph operators, moved out of the search
  subsystem. Unsupported operators raise an exception; this is not a full MPK
  task-graph persistence format.
- Header-only CUDA code formatting utilities, moved to `mirage/utils/codegen.h`.
- CUDA fingerprint implementations still embedded in common operator classes;
  graph construction disables fingerprint allocation by default. These are not a search service.
- MPK model adapters and their Triton/FlashInfer baselines.

Old search examples, including the workspace's `minimal_mpk_rms_norm.py`, require
`mpk-junyi`; they intentionally cannot run on this branch. Use the MPK RMSNorm
regression below for an example of registering dependent tasks and compiling them.
The old functionality remains available in Git history.

## Validation

```sh
python -m pip install pytest
python -m pytest tests/python -v
```

The GPU regression compiles and launches two dependent weighted BF16 RMSNorm tasks,
then compares both the intermediate tensor and final output with FP32 PyTorch
references rounded to BF16 (epsilon 1e-6). It skips explicitly on machines without CUDA.
CPU tests check graph serialization and import with Z3 blocked. Packaging tests check
native dependencies and headers required for MPK runtime compilation.

For wheel validation, build a wheel and run the same tests with the extracted wheel
on `PYTHONPATH`, with `MIRAGE_ROOT` unset. This ensures source headers do not hide
missing packaged headers.

Validated locally on an RTX A6000 (SM86), CUDA 13.1, Python 3.12:
clean native build; all six tests passed for both source and extracted wheel installs.
This does not validate full-model inference, multi-GPU execution, or other GPU architectures.
