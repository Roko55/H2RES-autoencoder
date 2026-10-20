# Wind diversity on the training-calibrated corpus

This comparison presents the validation-selected autoencoder, historical data
and the hybrid climate pipeline. All use the same corrected input corpus,
SHA256 `9a21558378d6359f308e3b7b7a3068d053195fa9123792c1d8aa9ac177101022`.
Training is 1980-2015, validation is 2016-2019, and reporting is 2020-2024.
National CF calibration targets end in 2015. The reporting period has been
inspected during development; it is not a new untouched test.

Each generator supplies 15 individual prior paths at sampling scale 1 and
seeds 101-103. The historical reference has five years. Reconstruction and
posterior-mean outputs are excluded. No reporting metric selects a displayed
path or checkpoint. The checkpoint was selected by validation.

| Annual wind-mean variability | Historical | Autoencoder | Hybrid |
|---|---:|---:|---:|
| Population standard deviation | 0.007968 | 0.002258 | 0.005439 |
| Fraction of historical variability | 100% | 28.3% | 68.3% |

The autoencoder has a better mean wind path score (0.302637 versus 0.345773)
and Wasserstein distance (0.022452 versus 0.025282), while the hybrid better
reproduces the reference 95th-percentile hourly ramps (0.085004 versus
0.057743; reference 0.090553). No generator is uniformly superior. The
application argument for wind resampling is annual diversity and ramps, not
a claim that the autoencoder cannot generate changing hourly output.

![Annual diversity](../../figures/corrected_annual_wind_comparison.png)

![Individual hourly paths](../../figures/corrected_wind_week.png)

The week is fixed to the first 168 hours, seed 101, first generated path and
hybrid member 31 from the frozen comparator. It was not chosen for its error
or visual appearance. Synthetic paths are alternatives, not forecasts of the
displayed historical week. The solver has not been rerun with this autoencoder.

Reproduce both figures with `python analysis/plot_corrected_wind.py`.
Use [corrected_study](../../corrected_study/README.md) for generation with the
selected checkpoint. Full channel metrics, annual means, figure data,
checkpoint selection and verification records accompany this folder.
