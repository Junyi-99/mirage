# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Run against an editable install or an extracted MPK wheel."""
from pathlib import Path
import subprocess
import mirage


def test_native_dependencies():
    result = subprocess.run(['ldd', mirage.core.__file__], text=True, capture_output=True, check=True)
    for removed in ('libz3', 'libabstract_subexpr', 'libformal_verifier'):
        assert removed not in result.stdout
    assert 'not found' not in result.stdout


def test_runtime_headers():
    from mirage.kernel import get_key_paths
    _, include, deps = get_key_paths()
    assert (Path(include) / 'mirage/persistent_kernel/persistent_kernel.cuh').is_file()
    assert (Path(deps) / 'cutlass/include/cutlass/cutlass.h').is_file()
    assert (Path(deps) / 'cutlass/tools/util/include/cutlass/util/host_tensor.h').is_file()
    assert (Path(deps) / 'json/include/nlohmann/json.hpp').is_file()
