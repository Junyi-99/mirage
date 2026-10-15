# Build MPK from source

Requires Linux, a CUDA toolkit with nvcc, a compatible C++17 compiler, CMake >= 3.24, and Python with PyTorch. CUDA libraries must be discoverable by the dynamic loader.

```sh
git submodule update --init --recursive
python -m pip install -e . -v
export MIRAGE_HOME="$PWD"
```

No Z3, Rust, or Cargo is needed. For a local architecture override, put
`set(CMAKE_CUDA_ARCHITECTURES 86)` (or your GPU architecture) in `build/config.cmake`.
MPK compiles its generated persistent kernel separately for the detected GPU.

See [MPK-only scope and validation](docs/mpk/mpk-only.md).
