#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${1:-/mnt/d/My-Work/Esophageal/New}"
CONFIG="${2:-configs/final_locked.yaml}"
MAIN_PYTHON="${RECAST_PYTHON:-/home/wiky/miniforge3/envs/qsar/bin/python}"

cd "$WORKSPACE"
"$MAIN_PYTHON" -m pip install -e .
bash scripts/run_full_pipeline.sh "$WORKSPACE" "$CONFIG"
bash scripts/setup_survpfn_challenger.sh "$WORKSPACE"
bash scripts/run_survpfn_challenger.sh "$WORKSPACE"
bash scripts/download_gencode50.sh "$WORKSPACE"
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" run-q1-extension
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" reannotate-external-omics
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-figures
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-supplementary-figures
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-q1-figures
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-manuscript
# Publication-toolkit figures (Fig 3-7, S1-S5); Fig 7 uses the locked IPCW
# decision-curve artifact from run-q1-extension, so this runs after it.
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-toolkit-figures
# Publication-ready tables (Table 1-6, Supplementary 1-10) formatted from make-manuscript.
"$MAIN_PYTHON" -m recast_surv.cli --config "$CONFIG" make-toolkit-tables

date --iso-8601=seconds > artifacts/final/logs/submission_pipeline_completed_at.txt
