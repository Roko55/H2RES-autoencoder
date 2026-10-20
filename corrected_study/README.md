# Training-calibrated autoencoder

This is the compact model selected by validation in the corrected-corpus study.
It has 167,343 parameters. Its checkpoint SHA256 is
`8df4288fe68496ced27c833468d6ee5ac9bd082b815a12a87d28bd2651441c1c`.
The checkpoint, normalization, source and pre-reporting selection record are
included. No raw weather or H2RES input corpus is redistributed.

From the repository root, after installing the root requirements:

```bash
python corrected_study/generate_profiles.py --run runs/compact_local_hires_prior_matching_seed17_1800s_retry01 --count 5 --seed 101 --output exports/profiles_01.npy
python -m unittest discover -s corrected_study -p test_study.py -v
```

Output is `[5, 8760, 3]` in solar CF, wind CF and degrees Celsius, with a
hash manifest. The output path is relative to the selected run and must be new.
No input weather file is needed for generation. Sampling scale 1 is the
recorded comparison setting; it is unrelated to ambient temperature.

The corrected training input has SHA256
`9a21558378d6359f308e3b7b7a3068d053195fa9123792c1d8aa9ac177101022`.
Training is 1980-2015; validation is 2016-2019; reporting is 2020-2024.
The reporting years have been inspected previously. Normalization and national
CF calibration use training data. Validation selects the checkpoint; reporting
scores do not select it. Re-training requires the exact source corpus under
`corrected_study/data/HR_data.parquet` and the recorded study protocol.

See [wind comparison](../results/training_only_20261020/README.md). The model
improves several metrics, but annual wind means remain narrowly distributed
for resource-case comparisons. This is not a claim of universal inferiority.
