#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/mnt/d/My-Work/Esophageal/New}"
config="${2:-configs/final_locked.yaml}"
python_bin="${RECAST_PYTHON:-/home/wiky/miniforge3/envs/qsar/bin/python}"
log_dir="$workspace/artifacts/final/logs"

mkdir -p "$log_dir"
cd "$workspace"

{
  date --iso-8601=seconds
  "$python_bin" --version
  "$python_bin" -m pip freeze
  sha256sum "$config"
  git -C third_party/SurvivalPFN rev-parse HEAD
  git -C third_party/SurvPFN rev-parse HEAD
} > "$log_dir/environment_manifest.txt"

find src tests configs scripts -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$log_dir/source_code_sha256.txt"

run_step() {
  local step="$1"
  "$python_bin" -m recast_surv.cli --config "$config" "$step" 2>&1 | tee "$log_dir/${step}.log"
}

run_step validate
run_step build-cohort
run_step build-reference
run_step build-features
run_step benchmark-transport
run_step benchmark
run_step audit-external
run_step validate-external-clinical
run_step analyze-biology
run_step make-figures
run_step make-supplementary-figures
run_step make-manuscript
# Publication-toolkit figures (Fig 3-7, S1-S5). Without run-q1-extension in this
# core pipeline, Fig 7 falls back to an in-figure Kaplan-Meier net benefit.
run_step make-toolkit-figures
# Publication-ready tables (Table 1-6, Supplementary 1-10) formatted from make-manuscript.
run_step make-toolkit-tables

"$python_bin" -m pytest 2>&1 | tee "$log_dir/tests.log"
date --iso-8601=seconds > "$log_dir/completed_at.txt"
