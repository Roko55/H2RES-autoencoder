# Implementation and data audit

## Findings

**The model preprocessing is isolated by split; the complete input-preparation chain is not a strict temporal holdout.** This distinction is material. The pinned Parquet hash matches the exact file used in the comparison. Its full-period solar mean 0.10306405 and wind mean 0.21055949 match the adjacent upstream calibration records. Wind calibration used IRENA 2004–2023 and solar calibration used 2019–2023. Both include post-training years; both also include 2020–2023 from the reporting period. The source preprocessing applies those factors before writing the shared Parquet.

This is an upstream look-ahead limitation for a temporal evaluation, not a gradient path from test observations into the autoencoder. It affects the common corpus used by both CVAE and hybrid. The current audit does not quantify its effect on absolute error or model ranking. Equal input data do not make a contaminated holdout independent. The reporting period had also already been inspected in the wider project. The published results are therefore labelled retrospective throughout.

A strict new study must start with pre-calibration source inputs, estimate every calibration parameter using permitted training information, then train/select/evaluate again. Clipping and rounded sidecar factors prevent a justified reconstruction of the original raw inputs by simply dividing the processed values. No existing result has been renamed as a newly independent test.

## What was checked

| Stage | Evidence and conclusion |
|---|---|
| Input identity | Exact SHA256, ordered unique hourly timestamps, all 1980–2024 years, leap-day removal and complete 8760-hour years |
| Split boundaries | Training 1980–2015, validation 2016–2019, reporting 2020–2024; annual windows are constructed only from the training tensor |
| Fitting statistics | Min/max after wind `log1p`, night mask, metric scales and wind thresholds use 1980–2015 only |
| Future-data perturbation | Synthetic validation/reporting values were changed substantially; training tensors, windows, scales, mask and thresholds remained identical |
| Generation isolation | With the same weights and seed, changing future weather leaves prior samples unchanged; only calendar conditions enter `sample()` |
| Numerical training | Finite losses and gradients, nonzero prior gradients, actual parameter updates, seeded replay, annual shapes and calendar-shifted masks |
| Inverse transforms | Wind uses `expm1`; reported weather references stay in raw source units and are not clipped to training bounds |
| Checkpoint integrity | Published weights and normalization match training records and frozen manifests; changed normalization, modified reports, partial epochs and path escapes are rejected |
| Metric definitions | Individual-path errors remain separate from smoothed ensemble means; pooled RMSE averages squared errors before taking the root; offshore baseline weights are explicit |
| Selection | Preserved validation eligibility and model selection precede the reported study evaluation; no new choice is made from the reporting scores |
| Upstream calibration | Uses post-training observations; strict-holdout mode correctly rejects the supplied corpus |

Training loads the complete validated file to build chronological partitions, but only the training partition enters optimization. Validation weather is used for the stated checkpoint-selection objective. That is normal validation use; it does not turn the reporting period into a pristine test.

## Annual wind variability

The selected temporal CVAE has 8,760 latent elements per annual path; the original reference used 128 global elements. It is not a comparison at equal compression. Overlapping annual windows yield 1,826 training examples from 36 independent calendar years, not 1,826 independent years.

The objective includes per-path mean/standard-deviation and sorted-distribution losses. Its prior-statistics term pairs a generated annual path with a training path without an annual climate-state condition. For the squared mean-matching term alone, independent pairing favours the central target mean and penalizes generated mean variability. This provides a plausible mechanism for narrow annual wind means; its causal contribution was not isolated in a new ablation. Temporal latent averaging and the absence of an explicit annual stochastic state can also matter.

This is a legitimate but restrictive design choice for annual-yield scenario diversity. Better annual diversity could require changing the objective, conditioning or latent structure; it does not logically require a larger network. The current code is not presented as globally optimal. The CPU 900 cases and successful consumer-laptop case remain visible, including where they outperform the selected GPU configuration on the reporting score.

The decoder's sigmoid and training min/max scaling also bound extrapolation beyond the training range. Validation reconstruction targets follow the original clipped normalization; primary reporting metrics use unclipped physical references. These details matter when interpreting extremes and distribution shifts. Good reconstruction is not treated as proof of good prior generation.

### Evidence scope

The measured limitation concerns the spread of annual wind means. Sequence memorization was not evaluated with a dedicated nearest-neighbour diagnostic. Reconstruction, prior generation and annual scenario diversity are distinct evaluation targets. The available evidence supports an application-specific comparison, without establishing a causal attribution to model size, training duration or memory capacity.

## Reproducibility and implementation

- A controlled same-CPU test reproduced the earlier replay discrepancy by changing only calendar-feature construction: float32 PyTorch trigonometry versus float64 NumPy trigonometry followed by float32 conversion. The maximum temperature-output difference was 0.001632 source units. The NumPy calendar reproduced archived arrays within float32 storage rounding. This isolates calendar-feature precision in the controlled replay. The maintained export now follows the dataset/evaluator calendar construction; see [the controlled check](results/calendar_precision_verification.json).
- Added maintained prior-only `evaluate.py` and `audit.py`. They verify both normalization and weights before loading, check train-only statistics and carry the upstream-calibration limitation into output metadata.
- Added synthetic split-isolation, artifact-tampering and metric-definition tests. No private weather file is required for the tests.
- Removed remote-task handling, experiment-queue PID exceptions, vendor discovery and machine policy dependencies from the maintained resource guard. It can stop only its own work. The exact measured runtime is preserved under `provenance/executed/`.
- Avoided loading the historical Colab/Optuna module when generating/evaluating a temporal model. The historical source remains byte-preserved; its original Colab entry point can install dependencies and is not the recommended generation entry point.
- Organized comparison code under `analysis/`, tests under `tests/`, and execution evidence under `provenance/`. Removed native hybrid models, duplicate hybrid runners, manuscript drafts and access/transfer scripts from this experiment's publication scope.

The executed model/data/loss implementations and all measured weights/results remain unchanged. Archived source hashes explain the distinction between original execution and maintained entry points. A frozen model is not silently retrained or substituted to make an audit pass.

## Remaining limits

The results do not establish calibrated extreme-event probabilities, physically coherent multivariate weather, global model optimality, or improved H2RES investment decisions. Spatial/source differences affect the historical H2RES comparison; its temperature entry is a training-climatology proxy. No downstream solver was rerun here. Data and code rights must be checked before changing the private repository's visibility.
