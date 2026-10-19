"""Evaluate prior samples using verified weights and their saved normalization.

Unlike reconstruction, prior sampling sees calendar conditions only. Held-out
weather is used after sampling to compute metrics. Existing results are never
overwritten; evaluation writes a new directory under exports/.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
import argparse
import io
import json
from audit import SELECTED_RUN, check_run, check_dataset, input_provenance, inside, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=SELECTED_RUN)
    parser.add_argument("--data", default="data/HR_data.parquet")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--freeze-manifest", default="provenance/test_freeze.json")
    parser.add_argument("--output", required=True, help="New directory under exports/")
    parser.add_argument("--strict-holdout", action="store_true")
    args = parser.parse_args()
    provenance = json.loads((ROOT / "data_provenance.json").read_text())
    scope = input_provenance(provenance, strict=args.strict_holdout)
    freeze = json.loads(inside(ROOT / args.freeze_manifest, ROOT).read_text())
    report, checkpoint, low, high, night = check_run(ROOT, args.run, freeze if args.split == "test" else None)
    if args.split == "test":
        if sha256(ROOT / "overnight_protocol.json") != freeze["protocol_sha256"]:
            raise ValueError("Protocol changed after the test freeze")
        for name, expected in freeze["evaluation_implementation_sha256"].items():
            if name not in ("study_runner.py", "study_models.py", "study_data.py", "study_metrics.py", "latent_prior.py"):
                raise ValueError("Unknown frozen source entry")
            if sha256(ROOT / name) != expected:
                raise ValueError("Original evaluation/model source changed after the freeze")
    destination = inside(ROOT / args.output, ROOT / "exports")
    if destination.exists():
        raise FileExistsError("Choose a new output directory; preserved results cannot be overwritten")
    from study_runtime import ResourceGuard
    guard = ResourceGuard("cpu", 300)
    import numpy as np
    import pandas as pd
    import torch
    from study_models import legacy_module, make_model
    from study_data import physical
    from study_metrics import path_metrics
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    tensors, info = check_dataset(ROOT / args.data, low, high, night)
    conditions = tensors[3] if args.split == "validation" else tensors[5]
    observed = info[args.split]
    years = list(range(2016, 2020)) if args.split == "validation" else list(range(2020, 2025))
    # Local temporal models do not need the historical Colab/Optuna module.
    legacy = legacy_module("cpu") if report["variant"].startswith("legacy_") else None
    model = make_model(report["variant"], legacy, night).eval()
    model.load_state_dict(torch.load(io.BytesIO(checkpoint.read_bytes()), map_location="cpu", weights_only=True))
    protocol = json.loads((ROOT / "overnight_protocol.json").read_text())
    rows, correlations, paths = [], [], {}
    with torch.no_grad():
        for seed in protocol["prior_seeds"]:
            guard.check()
            torch.manual_seed(seed)
            # No observed weather, encoder output, or annual target enters sample().
            paths[seed] = physical(model.sample(conditions, temperature=1.).numpy(), low, high)
            for index, year in enumerate(years):
                metrics, joint = path_metrics(observed[index], paths[seed][index], info["scales"], night,
                                              year, "prior", seed, 1., info["low_wind_threshold"], info["high_wind_threshold"])
                rows.extend(metrics)
                correlations.append(joint)
    guard.check()
    destination.mkdir(parents=True, exist_ok=False)
    for seed, values in paths.items():
        np.savez_compressed(destination / f"prior_seed{seed}.npz", values=values.astype(np.float32), years=years)
    pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(correlations).to_csv(destination / "cross_channel.csv", index=False)
    summary = {"status":"completed", "split":args.split, "years":years,
               "checkpoint_sha256":sha256(checkpoint), "normalization_sha256":report["normalization_sha256"],
               "selection_score_T1":float(np.mean([row["selection_component"] for row in rows])),
               "public_evaluator_sha256":sha256(__file__), "test_used_for_selection":False, **scope}
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
