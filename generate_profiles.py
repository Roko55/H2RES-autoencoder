"""Fail-closed CPU generation from an actual study checkpoint for H2RES."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="-1")
sys.path.insert(0, str(ROOT / "vendor"))
import argparse
import hashlib
import io
import json
import numpy as np
import torch
from study_models import legacy_module, make_model
from study_data import physical
from study_runtime import ResourceGuard


def calendar_condition():
    """Match evaluation: float64 trigonometry followed by float32 conversion.

    Direct float32 PyTorch phases can change samples slightly on the same CPU.
    """
    hours = np.arange(8760)
    values = np.stack((np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
                       np.sin(2 * np.pi * hours / 8760), np.cos(2 * np.pi * hours / 8760)))
    return torch.from_numpy(values.astype(np.float32))[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--latent-prior", choices=("standard", "aggregate"), default="standard")
    parser.add_argument("--output", default="generated_profiles.npy", help="New file inside this run")
    args = parser.parse_args()
    if not 1 <= args.count <= 128 or not 0 <= args.temperature <= 2:
        raise ValueError("Count must be 1-128 and temperature 0-2")
    run = (ROOT / args.run).resolve()
    run.relative_to((ROOT / "runs").resolve())
    destination = (run / args.output).resolve()
    destination.relative_to(run)
    if destination.exists() or destination.with_suffix(".json").exists() or destination.suffix != ".npy":
        raise FileExistsError("Choose a new .npy output within the run")
    guard = ResourceGuard("cpu", 180)
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    checkpoint_info = report.get("checkpoint", {})
    checkpoint = run / checkpoint_info.get("file", checkpoint_info.get("filename", "best_weights.pth"))
    payload = checkpoint.read_bytes()  # A missing checkpoint raises; never substitute zeros.
    checkpoint_hash = hashlib.sha256(payload).hexdigest()
    if not checkpoint_info.get("sha256") or checkpoint_info["sha256"] != checkpoint_hash:
        raise ValueError("Checkpoint must match its recorded SHA256")
    norm = run / "normalization.npz"
    if not norm.exists():
        norm = run / "norm_params.npz"
    if report.get("normalization_sha256") and hashlib.sha256(norm.read_bytes()).hexdigest() != report["normalization_sha256"]:
        raise ValueError("Normalization hash differs from the training record")
    with np.load(norm) as values:
        low, high = values["data_min"].reshape(1, 3, 1), values["data_max"].reshape(1, 3, 1)
        night = values["night_mask"] if "night_mask" in values else None
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Invalid normalization")
    torch.set_num_threads(1)
    variant = report.get("variant", "legacy_original")
    legacy = legacy_module("cpu") if variant.startswith("legacy_") else None
    model = make_model(variant, legacy, night).eval()
    model.load_state_dict(torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True))
    del payload
    fitted_mean, fitted_factor = None, None
    if args.latent_prior == "aggregate":
        if args.temperature != 1:
            raise ValueError("The fitted aggregate sampler has its own covariance; use temperature 1")
        record = json.loads((run / "latent_calibration.json").read_text(encoding="utf-8"))
        calibration = run / "latent_calibration.npz"
        if record["checkpoint_sha256"] != checkpoint_hash or record["calibration_sha256"] != hashlib.sha256(calibration.read_bytes()).hexdigest():
            raise ValueError("Invalid aggregate prior provenance")
        with np.load(calibration) as fitted:
            fitted_mean, fitted_factor = fitted["mean"], fitted["cholesky"]
    calendar = calendar_condition()
    torch.manual_seed(args.seed)
    generated = []
    with torch.no_grad():
        for _ in range(args.count):
            guard.check()
            if args.latent_prior == "aggregate":
                from latent_prior import sample_gaussian
                sample = sample_gaussian(model, calendar, fitted_mean, fitted_factor)
            else:
                sample = model.sample(calendar, args.temperature)
            generated.append(physical(sample.numpy(), low, high)[0].T.astype(np.float32))
    profiles = np.stack(generated)
    if not np.isfinite(profiles).all() or profiles.shape != (args.count, 8760, 3):
        raise FloatingPointError("Invalid generated profiles")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        np.save(output, profiles)
    manifest = {"shape": list(profiles.shape), "channel_order": ["solar", "wind", "temperature"],
                "seed": args.seed, "temperature": args.temperature, "checkpoint_sha256": checkpoint_hash,
                "latent_prior": args.latent_prior,
                "normalization_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "units": "Unmodified source units, wind inverse log1p; no postprocessing or ensemble averaging"}
    with destination.with_suffix(".json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
