# Autoencoder for annual H2RES climate profiles

This package supplies one validation-selected conditional variational
autoencoder (CVAE) for annual solar capacity factor, wind capacity factor and
ambient temperature. It is compared with historical data and the
[H2RES climate pipeline](https://github.com/Roko55/H2RES-climate-pipeline).

## Install and run

Use Python 3.11 and install `requirements.txt`. For CPU-only PyTorch, install
its official CPU wheel before the other requirements. From this directory:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s model -p test_study.py -v
python audit.py
python generate_profiles.py --run runs/selected --count 5 --seed 101 --output exports/profiles_01.npy
python analysis/plot_wind_comparison.py
```

Weights and normalization are included. No raw weather file or retraining is
needed for generation. Output has shape `[5, 8760, 3]` in solar CF, wind CF
and degrees Celsius. The output path is relative to `model/runs/selected/`;
use a new filename on reruns. The JSON sidecar records seed and hashes.
Sampling scale 1 is the evaluation setting, not ambient temperature.

## Application comparison

For low, average and high annual resource cases, the relevant property is
variation between annual wind yields. The autoencoder generates changing
hourly output, but its annual means are more concentrated than the historical
reference and the pipeline sample.

| Wind diagnostic | Historical reference | Autoencoder | Pipeline |
|---|---:|---:|---:|
| SD of annual mean CF | 0.007968 | 0.002258 | 0.005439 |
| Fraction of historical annual-mean SD | 100% | 28.3% | 68.3% |
| Mean 95th-percentile absolute hourly ramp | 0.090553 | 0.057743 | 0.085004 |

![Annual wind diversity](figures/annual_wind_comparison.png)

Historical data contain five years. Autoencoder inference generates 15 paths
with prior scale 1 and seeds 101-103; the pipeline comparison selects 15
members from its frozen ensemble with these seeds. Dots are individual
annual means, not ensemble means.
These descriptive comparisons do not estimate rare-event probabilities.

![Individual wind paths](figures/wind_week.png)

The first calendar week and sampling seed 101 are fixed before plotting.
Synthetic paths are alternatives, not forecasts of that historical week.

The autoencoder performs better on other diagnostics: mean wind path score
is 0.302637 versus 0.345773 and Wasserstein distance is 0.022452 versus
0.025282 (lower is better). The pipeline better represents annual wind
diversity and hourly ramps in this sample. Neither is uniformly superior;
these diagnostics do not demonstrate better H2RES investment outcomes.

## Data and reproducibility

The model has 167,343 parameters and uses the corpus identified by SHA256
`9a21558378d6359f308e3b7b7a3068d053195fa9123792c1d8aa9ac177101022`.
Training is 1980-2015, validation is 2016-2019 and reporting is 2020-2024.
National CF calibration and normalization use training data. Validation
selects the checkpoint; reporting scores do not select it. Reporting years
were previously inspected and are not a newly untouched test.

`model/` contains the executed model and generation implementation, weights
and selection records. `results/wind_comparison/` contains metrics, plotting
values and verification records. `analysis/` reproduces the figures.
The recorded training protocol is time-bounded; it does not establish
convergence or impossibility of other autoencoders.

Raw weather, asset and solver inputs are not redistributed. See [RIGHTS.md](RIGHTS.md).
