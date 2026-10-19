# Required local inputs

Raw input files are excluded from this experiment's commit. The existing project may already contain the national weather file under `H2RES_SingleZone_Myopic_v1.2/data/SC/HR_data.parquet`.

For the preserved experiment, copy that exact file here or pass its path to `audit.py --data` and `evaluate.py --data`. The expected SHA256 is `d92748ca6625d0092b073b9bc16345da960a04c42da29926da5b125a63ad122d`. Do not replace it with another file and retain the old results label.

The corpus's upstream calibration uses post-training IRENA years. This is documented in `../data_provenance.json`. Strict temporal validation requires rebuilding the upstream corpus and conducting a new study; the existing freeze intentionally does not certify that claim.

Comparison recomputation additionally requires `original_genco.csv` and `original_ncre.csv` with hashes in `../results/input_hashes.json`. Synthetic generated arrays under `../frozen_outputs/` are evaluation artifacts, not the raw weather file or a training wind pool.
