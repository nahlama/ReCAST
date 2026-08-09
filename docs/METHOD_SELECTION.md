# Method and small-data model selection

Decision date: 2026-07-21

Eligible-method literature cutoff: **2026-06-30 inclusive**. The cutoff is not
a publication deadline.

## Mid-2026 method audit

| Method | First release | Small-data relevance | Decision |
|---|---:|---|---|
| SurvivalPFN | 2026-05-15 | Prior-fitted, censoring-native, benchmarked on 61 datasets against 21 methods | Primary survival backbone |
| SurvPFN | 2026-06-03 | Zero-shot censoring-aware distributional PFN; 2–10 features | Mandatory contemporary challenger; vendored at a pinned commit |
| Tabular foundation models for survival | 2026-01-29 | Converts censored survival into time-indexed classification tasks | Secondary challenger if reproducible code is available |
| TabSA-BJ | 2026-06-02 | Training-free TFM AFT/Buckley-James formulation | Secondary methodological comparator |
| scSurv | 2025-12/2026 issue | Direct single-cell-reference/bulk survival model | Mechanistic comparator; high per-cell model complexity for 77 events |
| MMOSurv | 2025 | Few-shot multi-omics meta-learning | Not primary: cross-cancer multi-omics meta-training is absent here |
| CoxKAN | 2025 | Nonlinear interpretable survival functions | Challenger only; fitted from scratch in a small event set |

SurvivalPFN remains the primary backbone because it has the broadest reported
survival-specific benchmark among the executable PFNs. SurvPFN, the newer method
within the cutoff, was executed in a separate pinned Python 3.12 runtime on the
identical split manifest and locked external subset.

## Reference estimator

**Selected:** donor-hierarchical, HPA-weighted robust NNLS with iterative Huber
residual weighting.

**Rejected as primary:** entropic unbalanced optimal transport. It remains a
prespecified negative ablation and its artifacts are preserved under
`artifacts/default/development_history/uot_compact`.

The decision is evidence-based. In 500 mixtures per scenario, robust NNLS
state MAE was 0.0265 (clean), 0.0263 (platform shift), and 0.0250 (unknown
component); UOT MAE was 0.0583, 0.0594, and 0.0612. Robust NNLS also had lower
Jensen-Shannon divergence in every scenario and a higher unknown-score
correlation in the unknown-component test.

The state estimates are used primarily for interpretation. Prediction receives
compact balances, diversity, matched-reference support, reconstruction cosine,
and a normalized residual unknown-expression score. The score must not be
described as an unknown cell fraction.

## Survival model

**Advanced primary candidate:** clinical-only SurvivalPFN using the released
`shi-ang/SurvivalPFN` checkpoint. **Required comparator:** elastic-net Cox.

SurvivalPFN is suitable for this setting because it is prior-fitted on synthetic
censored-survival tasks and does not train a high-capacity neural network from
scratch on only 182 patients and 77 deaths. Its adapter operates on seven
clinical variables after training-fitted median imputation, constant filtering,
and scaling.

Prespecified challengers use identical splits: SurvPFN, elastic-net Cox,
XGBoost-AFT, and random survival forest. SurvPFN is limited to the seven-feature
clinical panel because its released model supports 2-10 inputs. Distributionally
robust Cox remains unimplemented and must not be listed as an executed comparison.

## Evidence and boundary

In the locked 10-repeat five-fold benchmark, clinical elastic-net Cox had the
highest mean Uno C (0.600), followed by SurvivalPFN (0.593), random survival
forest (0.592), and XGBoost-AFT (0.544). SurvivalPFN had a slightly better IBS
than Cox (0.224 versus 0.227). Thus the advanced model did not dominate the
classical comparator on the internal primary metric.

The eligible June-2026 SurvPFN challenger achieved mean Uno C 0.593 and IBS
0.238. Against SurvivalPFN, its paired Uno-C improvement was 0.00046 (95% CI
-0.0268 to 0.0330), but its paired IBS improvement was -0.0140 (95% CI -0.0212
to -0.0060). Against Cox, its paired Uno-C improvement was -0.0070 (95% CI
-0.0290 to 0.0173), and its paired IBS improvement was -0.0110 (95% CI -0.0187
to -0.0032). It therefore did not replace either locked leading comparator.

Adding ReCAST to SurvivalPFN reduced mean Uno C from 0.593 to 0.560 and worsened
IBS from 0.224 to 0.234. The paired Uno C change was -0.0328 (95% CI -0.0486
to -0.0192). Full ReCAST/pathway features produced Uno C 0.571 and IBS 0.230.
The omics representation therefore failed the confirmatory incremental gate.

The locked external test used the 60 GSE53625 patients absent from GSE53624.
Clinical-only SurvivalPFN achieved Uno C 0.613 (95% bootstrap CI 0.481–0.744),
Harrell C 0.598, and IBS 0.509 over 1–3 years. This supports only moderate,
uncertain discrimination and shows poor calibration. Elastic-net Cox was tuned
only within TCGA (penalizer 0.1, L1 ratio 0) and then scored on the same locked
patients: Uno C 0.609, Harrell C 0.582, and IBS 0.567. SurvivalPFN remained the
numerical leader, but the small difference does not prove superiority.

SurvPFN scored the same 60 external patients at Uno C 0.574 (95% bootstrap CI
0.433-0.708), Harrell C 0.544, and IBS 0.492. Its discrimination was lower than
both locked comparators. The lower external IBS does not rescue the method
because calibration remained poor and internal IBS was significantly worse.

## Allowed reporting

Allowed:

- “SurvivalPFN was the advanced prespecified small-sample candidate, while
  elastic-net Cox led the internal primary metric.”
- “The June-2026 SurvPFN challenger tied SurvivalPFN on internal Uno C but had
  worse internal prediction error and lower external discrimination.”
- “Robust NNLS was more accurate than UOT in the stated technical simulations.”
- “Single-cell-derived ReCAST features did not improve survival prediction in
  this cohort.”

Not allowed:

- “SurvivalPFN is the most accurate ESCA survival model.”
- “ReCAST improves patient prognosis prediction.”
- “The external cohorts independently replicated the model.”
- “The unknown score estimates an unknown-cell fraction.”

Primary sources for SurvivalPFN:

- Paper: https://arxiv.org/abs/2605.15488
- Code: https://github.com/rgklab/SurvivalPFN
- Checkpoint: https://huggingface.co/shi-ang/SurvivalPFN

Primary sources for SurvPFN:

- Paper: https://arxiv.org/abs/2606.04564
- Code: https://github.com/genepi-freiburg/SurvPFN
- Checkpoint: https://huggingface.co/samuelboehm/SurvPFN
