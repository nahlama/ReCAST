# ReCAST-Surv

Code and results accompanying the ReCAST-Surv.

ReCAST-Surv is a reproducible, leakage-audited framework for testing whether
donor-aware single-cell reference information improves survival modelling in a
small bulk-tumour cohort. It integrates TCGA-ESCA expression and censored
survival, cell-level GEO single-cell references, MSigDB 2026.1 pathways, HPA
protein evidence, and compact clinical variables.

The validated reference estimator is donor-hierarchical, HPA-weighted robust
NNLS. Entropic unbalanced optimal transport (UOT) is retained as a negative
ablation: it was less accurate in the full synthetic-mixture benchmark.
SurvivalPFN is the small-data backbone and elastic-net Cox is the mandatory
classical comparator; the eligible June-2026 SurvPFN challenger was executed in
a pinned Python 3.12 environment.

This is a trajectory model, or a cell–cell
interaction learner. The pipeline never treats TISCH2 aggregate archives as raw
cell matrices and never modifies source data.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/recast_surv/` | Python package: cohort build, reference build, feature transport, benchmark, external validation, biology, figures, manuscript tables |
| `tests/` | pytest suite covering metrics, models, transport, reannotation, figures, and table generation |
| `configs/` | `smoke.yaml` (fast dev), `default.yaml` (development), `final_locked.yaml` (frozen confirmatory protocol) |
| `scripts/` | End-to-end pipeline runners, GENCODE 50 download, isolated SurvPFN challenger setup/run |
| `results/` | Frozen outputs of the `final_locked` run (mirror of `artifacts/final`, large `.parquet` intermediates excluded) |
| `docs/` | Analysis protocol, method-selection record, methods audit table, source manifests |

### `results/` contents

- `results/benchmark/` — 50 outer test folds, 800 panel/model fold results, paired comparisons, locked split manifest
- `results/technical_validation/` — 500 synthetic mixtures per scenario (clean, platform-shift, unknown-component)
- `results/external_validation/` — locked GSE53625 clinical test, audit report, cohort overlap audit
- `results/challengers/survpfn/` — June-2026 challenger metrics, frozen requirements, environment manifest
- `results/biology/` — 500-replicate donor-bootstrap marker stability, exploratory gene/pathway survival
- `results/extension/` — donor-held-out projection validation, GENCODE 50 external reannotation, decision curves, histology subgroups, PROBAST+AI self-assessment
- `results/manuscript/` — Tables 1–6, Supplementary Tables S1–S10, TRIPOD+AI / PROBAST+AI reporting checklist
- `results/figures/` — main and supplementary figures (PNG/SVG) plus figure manifests
- `results/logs/` — per-stage run logs, environment manifest, source-code SHA-256 record
- `results/provenance/` — input validation manifest and source hashes

## Key verified numbers

- TCGA-ESCA: 182 primary tumours, 77 deaths, 40,967 genes.
- Single-cell reference: 216,332 cells, 67 patients, 965 donor/state profiles, 13,614 shared genes, 13 modelled states.
- Robust NNLS state MAE: 0.0265 / 0.0263 / 0.0250 (clean / platform-shift / unknown-component). UOT: 0.0583 / 0.0594 / 0.0612.
- Confirmatory 10 × 5-fold benchmark: clinical elastic-net Cox mean Uno C 0.600; SurvivalPFN 0.593 with slightly lower IBS (0.224 vs 0.227). Adding ReCAST features reduced SurvivalPFN Uno C to 0.560 (paired mean change −0.0328, 95% CI −0.0486 to −0.0192).
- June-2026 SurvPFN challenger: internal Uno C 0.593, IBS 0.238 — Uno C tied within paired uncertainty, IBS significantly worse than both comparators.
- Locked external clinical test (60 GSE53625 patients absent from GSE53624, 33 deaths): SurvivalPFN Uno C 0.613 (95% CI 0.481–0.744) vs 0.609 for elastic-net Cox; IBS 0.509 vs 0.567. Calibration remained inadequate — this is not evidence for a deployable model.
- GSE53625 is a SuperSeries containing all 119 GSE53624 patients; the two must not be reported as independent cohorts.
- Outcome-free biology: 447 stable state–gene pairs, 92 unique prominent genes. None of the 92 genes or 24 pathways passed BH FDR < 0.05 in clinical-adjusted exploratory survival analysis.

## Installation

```bash
python -m pip install -e .
```

Requires Python ≥ 3.11. Optional extras: `.[survivalpfn]` (torch + huggingface-hub),
`.[rsf]` (scikit-survival), `.[dev]` (pytest).

```bash
python -m pytest
```

## Input data

Raw inputs are **not** redistributed here — they are public and are pulled from
their original repositories. Expected layout at the analysis root, matching the
paths in `configs/final_locked.yaml`:

```
TCGA_ESCA_Metadata.csv
ESCA_vst_normalized_matrix.csv
TCGA_ESCA_STAR_Counts.csv
Data/GEO_raw/{GSE154763,GSE160269,GPL19748}/
Data/TISCH2/{ESCA_GSE154763,ESCA_GSE160269}/
Data/HPA/proteinatlas.tsv
Data/{h.all,c2.cp.reactome,c2.cp.kegg_medicus,c7.immunesigdb}.v2026.1.Hs.symbols.gmt
Data/GENCODE/v50/gencode.v50.transcripts.fa.gz
step8_geo_prepared/{GSE53624,GSE53625}/
third_party/SurvivalPFN/
```

Exact URLs, byte counts, and SHA-256 hashes for every downloaded file are in
`docs/source_manifests/`. GENCODE 50 can be fetched with
`bash scripts/download_gencode50.sh`; the SurvivalPFN vendor checkout with
`bash scripts/setup_survpfn_challenger.sh`.

## Reproducing the confirmatory run

```bash
bash scripts/run_submission_pipeline.sh
```

Or stage by stage:

```bash
recast-surv --config configs/final_locked.yaml validate
recast-surv --config configs/final_locked.yaml build-cohort
recast-surv --config configs/final_locked.yaml build-reference
recast-surv --config configs/final_locked.yaml build-features
recast-surv --config configs/final_locked.yaml benchmark-transport
recast-surv --config configs/final_locked.yaml audit-external
recast-surv --config configs/final_locked.yaml validate-external-clinical
recast-surv --config configs/final_locked.yaml analyze-biology
recast-surv --config configs/final_locked.yaml run-q1-extension
recast-surv --config configs/final_locked.yaml reannotate-external-omics
recast-surv --config configs/final_locked.yaml make-figures
recast-surv --config configs/final_locked.yaml make-supplementary-figures
recast-surv --config configs/final_locked.yaml make-q1-figures
recast-surv --config configs/final_locked.yaml make-manuscript
```

Outputs are written to `artifacts/final/`; `results/` in this repository is the
frozen copy of that directory. Use `configs/smoke.yaml`, or
`benchmark --outer-repeats 1`, for development runs — the 10-repeat benchmark is
confirmatory and should only be run under the frozen protocol.

The June-2026 challenger runs in its own pinned environment:

```bash
bash scripts/setup_survpfn_challenger.sh
bash scripts/run_survpfn_challenger.sh
```

It verifies the locked split and input hashes, executes all 50 outer folds plus
the non-overlapping external test, and writes an exact package freeze and
checkpoint SHA-256.

## Data availability and licensing

All primary data are public: TCGA-ESCA (GDC), GEO accessions GSE154763,
GSE160269, GSE53624, GSE53625, platform GPL19748, TISCH2, the Human Protein
Atlas, MSigDB 2026.1, and GENCODE 50. Each source retains its own licence and
terms of use; nothing in this repository redistributes them.
