#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
python -m topoverus.launch sft "$@" --execute
