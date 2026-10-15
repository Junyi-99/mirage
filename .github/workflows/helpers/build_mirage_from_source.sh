#!/bin/bash
# Script to build Mirage from source
set -xeuo pipefail

# Setup environment variables
# Use current directory if not in GitHub Actions
if [ -n "${GITHUB_WORKSPACE:-}" ]; then
  MIRAGE_ROOT="$GITHUB_WORKSPACE"
else
  MIRAGE_ROOT="$(pwd)"
fi
export MIRAGE_ROOT

# Detect CUDA path from environment or use default
if [ -n "${CUDA_TOOLKIT_PATH:-}" ]; then
  export CUDA_PATH="$CUDA_TOOLKIT_PATH"
elif [ -d "/usr/local/cuda" ]; then
  export CUDA_PATH="/usr/local/cuda"
fi

# Import environment variables setup script
# shellcheck source=.github/workflows/helpers/set_env.sh
source "$(dirname "$0")/set_env.sh"

cd "$MIRAGE_ROOT" || { echo "Error: Could not change directory to $MIRAGE_ROOT"; exit 1; }
mkdir -p build 
cd build || { echo "Error: Could not change directory to build"; exit 1; }

cmake .. \
-DCMAKE_C_COMPILER="$CC" \
-DCMAKE_CXX_COMPILER="$CXX"

# Build with multiple cores
make -j"$(nproc)"

# Install Mirage
cd "$MIRAGE_ROOT" || { echo "Error: Could not change directory to $MIRAGE_ROOT"; exit 1; }
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"
pip install -e .
