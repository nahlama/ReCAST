# ReCAST-Surv locked analysis protocol v1

Frozen: 2026-07-21

Eligible-method literature cutoff: 2026-06-30 inclusive.

## Scope

This protocol governs the full-data confirmatory run. It uses every validated
input listed in `Data/GEO_raw/SOURCE_MANIFEST.tsv`, the complete TCGA-ESCA VST
matrix and metadata, all four MSigDB 2026.1 collections, and HPA protein
evidence. Raw inputs are read-only.

## Fixed analysis

- Cohort: one primary tumour per TCGA patient; ESCC and EAC; OS time greater
  than or equal to one day.
- Reference: complete GSE160269 raw UMI data plus complete GSE154763 normalized
  cell-level data; donor-first profiles; no cell-count truncation.
- Reference estimator: HPA-weighted robust NNLS with five Huber iterations.
- Technical comparator: UOT, evaluated in 500 mixtures for each of clean,
  platform-shift, and unknown-component scenarios.
- Prediction panels: clinical, clinical plus pathways, clinical plus ReCAST,
  and full.
- Models: SurvivalPFN, elastic-net Cox, XGBoost-AFT, and random survival forest.
- Internal validation: 10 repeats of patient-level five-fold outer CV with
  four-fold inner tuning where applicable.
- Primary metric: Uno IPCW C-index. Safeguards: Harrell C, IPCW Brier/IBS,
  1/3/5-year dynamic AUC, and calibration intercept/slope.
- External test: only the 60 GSE53625 patients absent from GSE53624. External
  outcomes may not tune features, model parameters, horizons, or recalibration.
- External models: SurvivalPFN and TCGA-internally tuned elastic-net Cox.
- External uncertainty: 5,000 patient bootstrap draws.

## Claim boundary

Technical superiority applies only to the stated mixture scenarios. Survival
superiority requires consistent internal and external support plus acceptable
calibration. A negative ReCAST increment must be reported and cannot be rescued
by post-hoc feature or subgroup selection.

SurvPFN (2026-06-03) is an eligible contemporary challenger. Its upstream
Python 3.12 runtime is isolated from the locked WSL `qsar` environment and must
be reported separately until reproduced with the identical split manifest.

## Figures

All main figures are generated from saved artifacts as editable SVG. Text must
remain live, bold Times New Roman (`svg.fonttype: none`). PNG files are QA
renders only and are not manuscript masters.
