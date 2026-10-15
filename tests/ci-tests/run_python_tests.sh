#!/usr/bin/env bash
set -euo pipefail
MIRAGE_HOME=$(realpath "${BASH_SOURCE[0]%/*}/../..")
export MIRAGE_HOME
python -m pytest "$MIRAGE_HOME/tests/python" -v
