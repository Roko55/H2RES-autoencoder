"""Freeze exact newly reproduced checkpoints before their test evaluation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--selected-run", required=True, help="Selected using validation only")
    parser.add_argument("--output", default="study_reproduction/test_freeze.json")
    parser.add_argument("--validation-dir", help="Prior-only validation output; allowed for a single run")
    args = parser.parse_args()
    if args.selected_run not in args.runs:
        raise ValueError("Selected run must be among the frozen runs")
    if args.validation_dir and len(args.runs) != 1:
        raise ValueError("An explicit validation directory requires exactly one run")
    destination = (ROOT / args.output).resolve()
    destination.relative_to(ROOT.resolve())
    if destination.exists():
        raise FileExistsError("Never overwrite a test freeze; choose a new study directory")
    checkpoints = []
    for name in args.runs:
        run = (ROOT / name).resolve()
        run.relative_to((ROOT / "runs").resolve())
        report = json.loads((run / "report.json").read_text(encoding="utf-8"))
        from audit import check_run
        check_run(ROOT, name)
        validation_dir = (ROOT / args.validation_dir).resolve() if args.validation_dir else run / "study_validation"
        validation_dir.relative_to(ROOT.resolve())
        validation = json.loads((validation_dir / "summary.json").read_text(encoding="utf-8"))
        if validation.get("split") != "validation":
            raise ValueError("Freeze requires validation output, not reporting/test output")
        info = report["checkpoint"]
        checkpoint = run / info.get("file", info.get("filename", "best_weights.pth"))
        actual = sha(checkpoint)
        if actual != info["sha256"] or actual != validation["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed after validation")
        item = {"run": name, "checkpoint": checkpoint.name, "sha256": actual}
        normalization = run / "normalization.npz"
        if not normalization.exists():
            normalization = run / "norm_params.npz"
        if normalization.exists():
            normalization_hash = sha(normalization)
            if report.get("normalization_sha256") and normalization_hash != report["normalization_sha256"]:
                raise ValueError("Normalization changed after training")
            if validation.get("normalization_sha256", normalization_hash) != normalization_hash:
                raise ValueError("Validation used a different normalization")
            item["normalization_sha256"] = normalization_hash
        if (run / "latent_calibration.npz").exists():
            item["calibration_sha256"] = sha(run / "latent_calibration.npz")
        checkpoints.append(item)
    value = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "checkpoints": checkpoints,
             "selection": {"run": args.selected_run, "test_used_for_selection": False},
             "protocol_sha256": sha(ROOT / "overnight_protocol.json"), "test_based_tuning_allowed": False,
             "evaluation_implementation_sha256": {name: sha(ROOT / name) for name in
                 ("study_runner.py", "study_models.py", "study_data.py", "study_metrics.py", "latent_prior.py")}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2)
    print("Frozen", len(checkpoints), "checkpoints:", destination)


if __name__ == "__main__":
    main()
