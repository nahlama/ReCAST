#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-/mnt/d/My-Work/Esophageal/New}"
PYTHON="${SURVPFN_PYTHON:-/home/wiky/miniforge3/envs/recast-survpfn/bin/python}"

cd "$WORKSPACE"
exec "$PYTHON" scripts/run_survpfn_challenger.py "${@:2}"
