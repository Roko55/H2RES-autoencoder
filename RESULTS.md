# Reading the results

All tables refer to a shared, pre-calibrated retrospective corpus; see [AUDIT.md](AUDIT.md). The reporting period is 2020–2024/all 43,800 non-leap hours. Three sampling seeds produce five annual paths each. Original H2RES fixed inputs repeat the same calendar profile; its temperature is explicitly a training-climatology proxy. Onshore-only and original all-wind capacity weighting are separate cases.

## Quality measures

The composite criterion is the mean over channels and individual paths of:

`W1 / train_SD + 0.25 * abs(generated_SD / observed_SD - 1) + 0.25 * abs(generated_p99 - observed_p99) / train_SD + 0.5 * mean(abs(generated_ACF - observed_ACF))`

ACF lags are 1, 24 and 168 hours. Lower is better. This is the declared study criterion, not a universal quality threshold. `aggregate_individual_metrics.csv` also reports MAE, pooled RMSE, bias, W1, KS, p99 error, ACF errors and ramp errors. `individual_path_metrics.csv` retains path/year-level data, including wind-episode durations. Calendar-paired errors for stochastic weather are diagnostic rather than forecast skill.

`central_profile_metrics.csv` evaluates the mean of 15 generated paths separately. Smoothing cannot replace individual-path quality. `annual_mean_spread.csv` and `renewable_yield_index_spread.csv` quantify annual diversity; the latter uses the existing H2RES index of annual mean solar CF plus wind CF. Five observed years and 15 generated years support descriptive comparisons, not confident tail calibration.

## Column and model labels

In the main figure/table, **Historical data (reference)** means five source-data years (2020–2024), not the original H2RES model. **Autoencoder (CVAE)** means the single validation-selected joint generator. **Current pipeline (XGBoost + wind resampling)** means the default hybrid refit, with XGBoost for solar/temperature and seasonal block resampling for wind. Both generators supply 15 annual paths. The original fixed H2RES input profile is a distinct supplementary baseline. Internal CSV identifiers map to these labels as follows:

| CSV identifier | Reader-facing label |
|---|---|
| `observed_reporting` / observed reporting-year fields | Historical data (reference) |
| `selected_CVAE` | Autoencoder (CVAE) |
| `current_default_GPUfit_CPUpredict_seed42` | Current pipeline (XGBoost + wind resampling) |
| `original_H2RES_as_published_allwind` | Original fixed H2RES inputs (supplementary baseline) |

Chart height is SD across annual means. It is neither training loss nor an hourly error. The grey reference bar shows the spread among historical annual averages.

## Primary application comparison

The [README](README.md) fixes the validation-selected CVAE and the current-default hybrid refit. Its focus is the use of annual low/average/high resource cases in H2RES: annual wind-mean spread, renewable-yield spread and wind-episode durations. These diagnostics are an application-focused interpretation of existing results, not a new prespecified rejection rule. They were not used to replace the selected checkpoint or tune the model after reporting evaluation. The reproducible [focused table](results/application_comparison.csv) is derived solely from the preserved full tables by `analysis/present_application_comparison.py`.

The autoencoder's wind annual-mean SD is 14.0% of observed reporting-year SD, versus 74.8% for the default hybrid. Renewable-yield index SD is 0.001735 versus 0.007695 (observed 0.010971). Low/high wind episodes average 5.55/4.39 h for the autoencoder, 4.64/3.68 h for the hybrid and 3.18/2.19 h in observed data. Both generators have wind-event errors. Other diagnostics favour the autoencoder: its composite score is better, as is its wind lag-24 ACF error. The hybrid also has less annual solar/temperature spread than this autoencoder. Thus the focused view supports wind-driven annual resource diversity, not comprehensive statistical superiority or calibrated multivariate scenarios.

## Supplementary configurations and complete scores

These are separate recorded runs, not interchangeable results of the primary autoencoder. They remain available to show the scope and limits of the search; the model was not reselected using these reporting scores. Lower composite score is better.

| Recorded case | Composite score |
|---|---:|
| Primary validation-selected CVAE | 0.1950 |
| Same architecture, CPU 900 s budget | 0.1948 |
| Eight-hour temporal CVAE, consumer CPU | 0.1797 |
| Original-source CVAE, 100-epoch reference | 0.4978 |
| Primary current-default hybrid refit | 0.3064 |
| Saved production pipeline, first 15 paths | 0.3050 |
| Hybrid CPU refits, seeds 17 / 29 / 43 | 0.3084 / 0.3082 / 0.3065 |
| Original fixed H2RES inputs, all-wind + temperature proxy | 0.8177 |
| Original inputs, onshore sensitivity + temperature proxy | 0.7283 |

The original-source CVAE is a new reference run, not a certified historical checkpoint. The full-precision source is [aggregate_scores.csv](results/aggregate_scores.csv); all annual/channel metrics remain in the linked CSVs. No measured result was removed or reassigned. Smaller CPU cases can score well, so neither a RAM-only failure nor the necessity of longer training/larger models follows from this comparison.

## Interpreting clustered annual means

Insufficient generative diversity is the demonstrated output limitation. It does not demonstrate overfitting or exact repetition of historical paths. A dedicated memorization test and a controlled objective/latent-structure ablation were not performed. For independently paired generated and target annual means G and T, the squared matching term has expectation `Var(G) + Var(T) + (E[G] - E[T])^2`. Taken alone, it penalizes generated mean variance. That gives a plausible model/objective explanation, not a measured attribution of the complete loss's effect. The implementation audit therefore does not claim that all viable autoencoder designs were exhausted or that a hardware constraint alone prevented improvement.

## Resources and replay

`training_resources.csv` distinguishes tree fitting, full calibration, preparation and generation. GPU timings refer to the QuadroP2000; CPU runs use one thread. Peak process RAM includes imported libraries and is not total machine memory or VRAM. Native XGBoost GPU peaks were not measured comparably. Hardware and software records remain in the run reports.

After a one-year warmup, the selected CVAE generated 15 years in 2.386 s on the same faculty CPU where the default hybrid took 52.285 s. Load/warmup costs are separate. An earlier replay discrepancy of about 0.00163 temperature-source units was subsequently reproduced on the same CPU by changing only calendar precision. The maintained export now uses NumPy float64 calendar calculations followed by float32 conversion, matching the data evaluator. Original reporting arrays and measured timing records were unchanged.

The public prior evaluator writes new outputs and records its own source hash. Its CPU replay is checked against the archived score within the stated numerical tolerance. These checks establish implementation replay on the recorded corpus, not an independent scientific replication with new data.
